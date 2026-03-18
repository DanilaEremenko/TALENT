import math
import sys

from torch.nn.functional import one_hot

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
from catkernel.base_kernel import CkStateException
from catkernel.nw_kernel import NWScikit
from hyperparams.hp_nw import get_best_mname


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

        x_B_l = []
        if self.N is not None:
            n_num_f = self.N['train'].shape[1]
            x_B_l.append((self.N['train']))
        else:
            n_num_f = 0

        if self.C is not None:
            x_B_l.append((self.C['train']))
            n_cat_f = self.C['train'].shape[1]
            cat_ids = list(range(n_num_f, n_num_f + n_cat_f))
        else:
            cat_ids = []

        x_B = torch.concat(x_B_l, dim=1).clone()
        y_B = self.y['train'].clone()

        if self.args.use_float:
            x_B = x_B.float()
            y_B = y_B.float()

        from hyperparams import hp_nw
        problem_mode = 'reg' if self.D.is_regression else 'clf'
        fit_y = problem_mode == 'reg'
        meta_model = hp_nw.get_models_hparams(
            problem_mode=problem_mode,
            model_name=get_best_mname(fit_y=fit_y)
        )
        common_params = {
            'act_fn': None,
            'dist_model_biases': False,
            'n_layers': None,
            'rbp_init_max': 1000 if self.args.model_type == 'nw_init_1000' else None,
            'optimized_cdist': True
        }

        self.model_sk_wrapper = NWScikit(
            **model_config,
            tmp_dir=None,
            cat_ids=cat_ids,
            **meta_model.common_params,
            **common_params
            # **{key: val.to_lamda_d() for key, val in h_params.random_params.items()}
        )

        if problem_mode == 'clf':
            y_B = one_hot(y_B.long()).float()
        elif problem_mode == 'reg':
            y_B = y_B.unsqueeze(1) if y_B.ndim == 1 else y_B
        else:
            raise ValueError(problem_mode)

        self.model_sk_wrapper._model = self.model_sk_wrapper.get_model_instance(
            X=x_B, y=y_B
        )
        self.model = self.model_sk_wrapper._model
        self.x_B = self.model.x_B
        self.model.x_B = None

        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

        self.grad_exploded = False

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
        self.optimizers, self.schedulers = self.model_sk_wrapper.get_optimizers_and_schedulers(
            n_batches=math.ceil(len(self.y['train']) / self.args.batch_size)
        )
        self.train_size = self.N['train'].shape[0] if self.N is not None else self.C['train'].shape[0]
        self.train_indices = torch.arange(self.train_size, device=self.args.device)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return

        time_cost = 0
        self.validate(0)
        self.val_count = 0
        for epoch in range(self.args.max_epoch):
            tic = time.time()
            try:
                self.train_epoch(epoch)
                self.validate(epoch)
            except CkStateException:
                self.grad_exploded = True
                break
            elapsed = time.time() - tic
            time_cost += elapsed
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        self.last_epoch = epoch
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

                X = X if isinstance(X, torch.Tensor) else torch.concat([x for x in X if X is not None], dim=1)

                if self.args.use_float:
                    X = X.float()

                pred = self.model(
                    X=X,
                    x_B=self.x_B,
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

            x_l = []
            if self.N is not None:
                x_l.append(self.N['train'][batch_idx])

            if self.C is not None:
                x_l.append(self.C['train'][batch_idx])

            X = torch.concat(x_l, dim=1)

            y = self.y['train'][batch_idx]

            if self.args.use_float:
                X = X.float()

            pred = self.model(
                X=X,
                x_B=self.x_B,
                indices=batch_idx if self.model_sk_wrapper.lvo else None
            ).squeeze(-1)

            loss = self.criterion(pred, y)

            tl.add(loss.item())
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.backward()

            for optimizer in self.optimizers:
                optimizer.step()

            del loss
            i += 1

        tl = tl.item()
        self.trlog['train_loss'].append(tl)

    def validate(self, epoch):
        # print('best epoch {}, best val res={:.4f}'.format(
        #     self.trlog['best_epoch'],
        #     self.trlog['best_res']))

        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label = [], []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                X = torch.concat(X, dim=1) if isinstance(X, list) else X
                if self.args.use_float:
                    X = X.float()
                pred = self.model(
                    X=X,
                    x_B=self.x_B,
                    indices=None,
                ).squeeze(-1)

                test_logit.append(pred)
                test_label.append(y)

        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)

        vl = self.criterion(test_logit, test_label).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal

        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if self.trlog['best_res'] is None or measure(vres[0], self.trlog['best_res']) or epoch == 0:
            # sys.stderr.write(f'trial upd metric {dict(zip(metric_name, vres, strict=True))["RMSE"]}')
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
