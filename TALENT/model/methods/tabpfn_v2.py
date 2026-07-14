from TALENT.model.methods.base import Method
import torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from TALENT.model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_label_process
)
from TALENT.model.lib.tabpfn_v2.tabpfn.utils import _fix_dtypes, validate_X_predict
import time


class TabPFNMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.normalization == 'none')
        assert(args.cat_policy == 'indices')
        assert(args.num_policy == 'none')
        assert(args.tune != True)


    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.criterion = F.cross_entropy if  not self.is_regression else F.mse_loss
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']


    def construct_model(self, model_config = None,cat_indices=[]):
        if self.is_regression:
            from TALENT.model.models.tabpfn_v2 import TabPFNRegressor
            self.model = TabPFNRegressor(
                model_path = "./TALENT/model/models/models_tabpfn/tabpfn-v2-regressor.ckpt",
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = 8,
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )
        else:
            from TALENT.model.models.tabpfn_v2 import TabPFNClassifier
            self.model = TabPFNClassifier(
                model_path = "./TALENT/model/models/models_tabpfn/tabpfn-v2-classifier.ckpt",
                device = self.args.device,
                random_state = self.args.seed,
                n_estimators = 4,
                ignore_pretraining_limits = True,
                categorical_features_indices = cat_indices
            )


    def fit(self, data, info, train = True, config = None):
        N, C, y = data
        # if self.D is None:
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        self.data_format(is_train = True)
        
        sampled_Y = self.y['train']
        cat_indices = []
        if self.N is not None and self.C is not None:
            sampled_X = np.concatenate((self.N['train'],self.C['train']),axis=1)
            cat_indices = [i for i in range(self.N['train'].shape[1],self.N['train'].shape[1]+self.C['train'].shape[1])]
        elif self.N is None and self.C is not None:
            sampled_X = self.C['train']
            cat_indices = [i for i in range(self.C['train'].shape[1])]
        else:
            sampled_X = self.N['train']
        sample_size = self.args.config['general']['sample_size']
        self.sampled_X = sampled_X
        self.sampled_Y = sampled_Y
        self.construct_model(cat_indices=cat_indices)
        self.model.fit(sampled_X,sampled_Y,sample_size)
        self.fit_time = 0  # general model does not require fitting


    def predict(self, data, info, model_name, do_eval_stats=False):
        N, C, y = data
        self.data_format(False, N, C, y)
        if self.N_test is not None and self.C_test is not None:
            Test_X = np.concatenate((self.N_test, self.C_test), axis=1)
        elif self.N_test is None and self.C_test is not None:
            Test_X = self.C_test
        else:
            Test_X = self.N_test
        
        tic = time.time()
        if self.is_regression:
            test_logit = self.model.predict(Test_X)
        else:
            test_logit = self.model.predict_proba(Test_X)
        self.predict_time = time.time() - tic

        if do_eval_stats:
            self.eval_stats = dict(eval_stats_l=[self._compute_eval_stats(Test_X)])
            self.eval_stats |= dict(predict_time=self.predict_time)

        test_label = self.y_test
        vl = self.criterion(torch.tensor(test_logit), torch.tensor(test_label)).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        
        # Denormalize regression predictions back to original scale
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            test_logit = test_logit * self.y_info['std'] + self.y_info['mean']
        print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit

    def _compute_eval_stats(self, Test_X):
        """IG attributions + a KMeans clustering of TabPFN's own test-token
        embeddings, mirroring tabm.py/modernNCA.py's do_eval_stats output.
        TabPFN needs its own version of this (rather than reusing their
        return_embs=True convention) because:
        - predict()/predict_proba() run the transformer forward pass inside
          torch.inference_mode() (see InferenceEngineCachePreprocessing.
          iter_outputs), which blocks autograd entirely, so IG can't hook
          into that call path;
        - the sklearn wrapper exposes no return_embs-style argument;
          embeddings only appear when the *raw* transformer is called with
          only_return_standard_out=False (out["test_embeddings"]/
          out["standard"], see PerFeatureTransformer._forward);
        - the deployed ensemble's own preprocessors (engine.preprocessors,
          e.g. default_classifier_preprocessor_configs()) do NOT preserve
          the raw feature space: "quantile_uni_coarse" + append_original=True
          + global_transformer_name="svd" turns 16 raw columns into 41 (16
          original + 16 quantile-transformed + SVD components), which is
          incomparable to imps_T_true (computed in the 16-raw-feature space
          by get_common_xai_clust_stats). So this fits its OWN throwaway,
          no-op ensemble member (PreprocessorConfig("none", ...), no
          append/SVD/polynomial/fingerprint features) purely for this
          explanation pass — column-count- and column-order-preserving
          (only per-column standardization, which doesn't change relative
          feature importance), never used for the actual prediction/metrics.
        Uses only the first args.batch_size test rows (there is no
        dataloader here — predict() scores the whole test set in one call —
        so this stands in for tabm.py/modernNCA.py's "first batch only"
        do_eval_stats convention)."""
        from utils_xai_local.ig import explain_nn_ig
        from TALENT.model.lib.tabpfn_v2.tabpfn.preprocessing import (
            ClassifierEnsembleConfig,
            PreprocessorConfig,
            RegressorEnsembleConfig,
            fit_preprocessing,
        )

        engine = self.model.executor_
        if not hasattr(engine, 'preprocessors'):
            # Only fit_mode="fit_preprocessors" (InferenceEngineCachePreprocessing,
            # the default used by construct_model) is supported here.
            return dict(ig_values=[], cluster_test=[])

        cat_ix = engine.cat_ixs[0]
        device = self.model.device_

        X = validate_X_predict(Test_X, self.model)
        X = _fix_dtypes(X, cat_indices=self.model.categorical_features_indices)
        X = self.model.preprocessor_.transform(X)
        X = X[: self.args.batch_size]

        # Same outer (ordinal-encoding) transform used at fit time, replayed
        # on the exact training data — reproduces the raw, pre-ensemble
        # X_train/y_train that create_inference_engine originally received
        # (not stored on engine itself, only the already-preprocessed
        # per-ensemble-member versions are).
        X_train_raw = self.model.preprocessor_.transform(
            _fix_dtypes(self.sampled_X, cat_indices=self.model.categorical_features_indices)
        )
        no_op_preprocess_config = PreprocessorConfig(
            "none", categorical_name="numeric", subsample_features=-1,
        )
        if self.is_regression:
            y_train_raw = (self.sampled_Y - self.model.y_train_mean_) / self.model.y_train_std_
            no_op_config = RegressorEnsembleConfig(
                preprocess_config=no_op_preprocess_config,
                add_fingerprint_feature=False, polynomial_features="no",
                feature_shift_count=0, feature_shift_decoder=None, subsample_ix=None,
                target_transform=None,
            )
        else:
            y_train_raw = self.model.label_encoder_.transform(self.sampled_Y)
            no_op_config = ClassifierEnsembleConfig(
                preprocess_config=no_op_preprocess_config,
                add_fingerprint_feature=False, polynomial_features="no",
                feature_shift_count=0, feature_shift_decoder=None, subsample_ix=None,
                class_permutation=None,
            )
        [(_, preprocessor, X_train_np, y_train_np, noop_cat_ix)] = fit_preprocessing(
            configs=[no_op_config], X_train=X_train_raw, y_train=y_train_raw,
            random_state=0, cat_ix=cat_ix, n_workers=1, parallel_mode='block',
        )

        transformer = engine.model.to(device)
        if engine.force_inference_dtype is not None:
            transformer = transformer.type(engine.force_inference_dtype)
        # The preceding predict()/predict_proba() call leaves save_peak_mem_factor
        # set on every layer (via MemoryUsageEstimator.reset_peak_memory_if_required),
        # which asserts grad must be disabled whenever it's active — but this
        # method needs grad enabled for IG, so clear it before reusing the
        # same transformer instance here.
        transformer.reset_save_peak_mem_factor(None)
        try:
            X_train_t = torch.as_tensor(X_train_np, dtype=torch.float32, device=device)
            y_train_t = torch.as_tensor(y_train_np, dtype=torch.float32, device=device)
            X_test_t = torch.as_tensor(preprocessor.transform(X).X, dtype=torch.float32, device=device)
            n_train = len(y_train_t)

            def forward(x_test_2d):
                X_full = torch.cat([X_train_t, x_test_2d], dim=0).unsqueeze(1)
                out = transformer(
                    None, X_full, y_train_t,
                    only_return_standard_out=False,
                    categorical_inds=noop_cat_ix,
                    single_eval_pos=n_train,
                )
                return out['standard'].squeeze(1), out['test_embeddings'].squeeze(1)

            def model_fn(x_test_2d):
                logits, _ = forward(x_test_2d)
                if self.is_regression:
                    # Raw decoder output is bar-distribution bin logits, not a
                    # scalar prediction — decode to the distribution mean (a
                    # differentiable softmax + weighted sum, see
                    # FullSupportBarDistribution.mean) so IG attributes
                    # w.r.t. the actual predicted value, same as tabm.py/
                    # modernNCA.py attributing w.r.t. their scalar output.
                    return self.model.renormalized_criterion_.mean(logits).unsqueeze(1)
                return logits

            with torch.no_grad():
                _, embs = forward(X_test_t)

            with torch.enable_grad():
                n_targets = 1 if self.is_regression else model_fn(X_test_t).shape[1]
                ig_values = [
                    explain_nn_ig(
                        X_train=X_test_t, X_test=X_test_t,
                        model=model_fn, target=cls,
                    ).detach().cpu().numpy().tolist()
                    for cls in range(n_targets)
                ]
        finally:
            engine.model = transformer.cpu()

        return dict(
            ig_values=ig_values,
            cluster_test=KMeans(n_clusters=3).fit_predict(embs.detach().cpu().numpy()).tolist(),
        )
