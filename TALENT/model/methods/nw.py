from TALENT.model.methods.base import Method
import time
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from TALENT.model.utils import (
    Averager
)
from typing import Optional
from TALENT.model.lib.data import (
    Dataset
)
from catkernel.nw_kernel import NwKernelModel


def make_random_batches(
        train_size: int, batch_size: int, device: Optional[torch.device] = None
):
    permutation = torch.randperm(train_size, device=device)
    batches = permutation.split(batch_size)
    # Below, we check that we do not face this issue:
    # https://github.com/pytorch/vision/issues/3816
    # This is still noticeably faster than running randperm on CPU.
    # UPDATE: after thousands of experiments, we faced the issue zero times,
    # so maybe we should remove the assert.
    assert torch.equal(
        torch.arange(train_size, device=device), permutation.sort().values
    )
    return batches  # type: ignore[code]


class NwMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert (args.cat_policy == 'tabr_ohe')
        assert (args.num_policy == 'none')

    def construct_model(self, model_config=None):
        if model_config is None:
            model_config = self.args.config['model']
        cat_size = sum([len(c) for c in self.cat_encoder.categories_]) if self.cat_encoder is not None else 0
        self.model = NwKernelModel(
            **model_config,
            in_size=self.d_in + cat_size,
            out_size=self.d_out,
            device=self.args.device,
            kernel_fit_background=self.D.is_regression,
            problem_mode='reg' if self.D.is_regression else 'clf',
            cat_ids=None
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def fit(self, data, info, train=True, config=None):
        N, C, y = data
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
            self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features

            self.data_format(is_train=True)
        if config is not None:
            self.reset_stats_withconfig(config)
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.config['training']['lr'],
            weight_decay=self.args.config['training']['weight_decay']
        )
        self.train_size = self.N['train'].shape[0] if self.N is not None else self.C['train'].shape[0]
        self.train_indices = torch.arange(self.train_size, device=self.args.device)
        self.context_size = 96
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return

        time_cost = 0
        for epoch in range(self.args.max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            elapsed = time.time() - tic
            time_cost += elapsed
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        self.fit_time = time_cost

    def predict(self, data, info, model_name):
        N, C, y = data
        self.model.load_state_dict(
            torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()

        self.data_format(False, N, C, y)

        test_logit, test_label = [], []

        tic = time.time()
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None

                x_B_num = self.N['train'] if self.N is not None else None
                x_B_cat = self.C['train'] if self.C is not None else None
                y_B = self.y['train']

                if self.args.use_float:
                    X_num = X_num.float() if X_num is not None else None
                    X_cat = X_cat.float() if X_cat is not None else None
                    x_B_num = x_B_num.float() if x_B_num is not None else None
                    x_B_cat = x_B_cat.float() if x_B_cat is not None else None
                    if self.is_regression:
                        y_B = y_B.float()

                X = torch.concat([X_num, X_cat], dim=1) if X_cat is not None else X_num
                x_B = torch.concat([x_B_num, x_B_cat], dim=1) if X_cat is not None else x_B_num

                pred = self.model(
                    X=X,
                    x_background=x_B,
                    y_background=y_B.unsqueeze(1),
                    indices=None,
                ).squeeze(-1)

                test_logit.append(pred)
                test_label.append(y)

        self.predict_time = time.time() - tic

        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)

        vl = self.criterion(test_logit, test_label).item()

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # FIX: Denormalize regression predictions
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']

        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))

        return vl, vres, metric_name, test_logit

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        i = 0
        for batch_idx in make_random_batches(self.train_size, self.args.batch_size, self.args.device):
            self.train_step = self.train_step + 1

            X_num = self.N['train'][batch_idx] if self.N is not None else None
            X_cat = self.C['train'][batch_idx] if self.C is not None else None
            y = self.y['train'][batch_idx]

            x_B_num = self.N['train'] if self.N is not None else None
            x_B_cat = self.C['train'] if self.C is not None else None
            y_B = self.y['train']
            if self.args.use_float:
                X_num = X_num.float() if X_num is not None else None
                X_cat = X_cat.float() if X_cat is not None else None
                x_B_num = x_B_num.float() if x_B_num is not None else None
                x_B_cat = x_B_cat.float() if x_B_cat is not None else None
                if self.is_regression:
                    y_B = y_B.float()
                    y = y.float()
            X = torch.concat([X_num, X_cat], dim=1) if X_cat is not None else X_num
            x_B = torch.concat([x_B_num, x_B_cat], dim=1) if X_cat is not None else x_B_num
            pred = self.model(
                X=X,
                x_background=x_B,
                y_background=y_B.unsqueeze(1),
                indices=batch_idx
            ).squeeze(-1)

            loss = self.criterion(pred, y)

            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i - 1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
            i += 1

        tl = tl.item()
        self.trlog['train_loss'].append(tl)

    def validate(self, epoch):
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'],
            self.trlog['best_res']))

        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None

                x_B_num = self.N['train'] if self.N is not None else None
                x_B_cat = self.C['train'] if self.C is not None else None
                y_B = self.y['train']
                if self.args.use_float:
                    X_num = X_num.float() if X_num is not None else None
                    X_cat = X_cat.float() if X_cat is not None else None
                    x_B_num = x_B_num.float() if x_B_num is not None else None
                    x_B_cat = x_B_cat.float() if x_B_cat is not None else None
                    if self.is_regression:
                        y_B = y_B.float()

                x_B_num = self.N['train'] if self.N is not None else None
                x_B_cat = self.C['train'] if self.C is not None else None
                x_B = torch.concat([x_B_num, x_B_cat], dim=1) if X_cat is not None else x_B_num
                y_B = self.y['train']

                pred = self.model(
                    X=X,
                    x_background=x_B,
                    y_background=y_B.unsqueeze(1),
                    indices=None,
                ).squeeze(-1)

                test_logit.append(pred)
                test_label.append(y)

        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)

        vl = self.criterion(test_logit, test_label).item()

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))
