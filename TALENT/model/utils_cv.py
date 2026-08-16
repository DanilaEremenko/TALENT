"""Cross-validation hyper-parameter tuning for LAMDA-TALENT datasets."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

from catkernel.talent import utils as talent_utils


class _CVFolds(list):
    """List of folds compatible with the ordinary tuner data interface."""

    def __getitem__(self, index):
        # The common tuner uses train_val_data[2]["train"] to constrain
        # model-specific search spaces (currently KNN n_neighbors).  Expose
        # the first fold's targets for that read while keeping normal list
        # iteration for CV training.
        if index == 2:
            return super().__getitem__(0)[2]
        return super().__getitem__(index)


class _CVMethod:
    """Run one LAMDA method independently on every persisted CV fold."""

    def __init__(self, method_factory, args, is_regression):
        self._method_factory = method_factory
        self._args = args
        self._is_regression = is_regression
        self.trlog = {}

    def fit(self, folds, info, train=True, config=None):
        fold_scores = []
        fold_logs = []
        for fold_i, fold_data in enumerate(folds):
            fold_args = copy.deepcopy(self._args)
            fold_args.save_path = str(
                Path(self._args.save_path) / f"cv_fold_{fold_i}"
            )
            Path(fold_args.save_path).mkdir(parents=True, exist_ok=True)
            # A CV fold directory is reused by consecutive Optuna trials.
            # Never let a trial read a checkpoint produced by a previous
            # trial if this trial failed before writing its own checkpoint.
            for checkpoint_name in (
                f'best-val-{fold_args.seed}.pth',
                f'best-val-{fold_args.seed}.pt',
                f'best-val-{fold_args.seed}.pkl',
                f'best-val-{fold_args.seed}.joblib',
                f'epoch-last-{fold_args.seed}.pth',
                'trlog',
            ):
                checkpoint_path = Path(fold_args.save_path) / checkpoint_name
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            method = self._method_factory(fold_args, self._is_regression)
            N, C, y = fold_data
            # Persisted val is the out-fold. Only fit_data receives train as
            # val for early stopping.
            out_X = N["val"]
            out_y = y["val"]
            fit_data = (
                {"train": N["train"], "val": N["train"]},
                C,
                {"train": y["train"], "val": y["train"]},
            )
            out_data = (
                {"test": out_X},
                C,
                {"test": out_y},
            )
            val_stats = [
                {
                    "keys": list(f[0].keys()),
                    "shape": getattr(f[0].get("val"), "shape", None),
                    "min": float(np.nanmin(f[0]["val"])),
                    "max": float(np.nanmax(f[0]["val"])),
                }
                for f in folds
            ]
            assert not all([(f[0]['val'] == 0).all() for f in folds]), val_stats
            method.fit(fit_data, info, train=train, config=copy.deepcopy(config))
            if getattr(method, 'grad_exploded', False):
                self.trlog['best_res'] = None
                self.trlog['fold_scores'] = []
                sys.stderr.write(f'CV fold {fold_i}: NW exploded during fit\n')
                return self
            checkpoint_paths = [
                Path(fold_args.save_path) / f'best-val-{fold_args.seed}{suffix}'
                for suffix in ('.pth', '.pt', '.pkl', '.joblib')
            ]
            assert any(path.is_file() for path in checkpoint_paths), checkpoint_paths
            prediction = method.predict(
                out_data,
                info,
                model_name=self._args.evaluate_option,
            )
            # Deep methods return (loss, metrics, names, predictions), while
            # classical methods return (metrics, names, predictions).
            metrics = prediction[1] if len(prediction) == 4 else prediction[0]
            score = metrics[0]
            fold_scores.append(float(score))
            fold_logs.append(method.trlog)

        assert fold_scores is not None, fold_scores

        self.trlog['best_res'] = np.mean(fold_scores)

        for key in (
                'best_n_cl_alive', 'best_cl_T_probs_max_mean',
                'best_cl_energy_minmax_ratio', 'best_cl_energy_min_share',
                'last_n_cl_alive', 'last_cl_T_probs_max_mean',
                'last_cl_energy_minmax_ratio', 'last_cl_energy_min_share',
        ):
            values = [log[key] for log in fold_logs if key in log]
            if values:
                self.trlog[key] = float(np.asarray(values, dtype=float).mean())

        sys.stderr.write(f"trial: {fold_scores}\n")
        self.trlog['fold_scores'] = fold_scores
        return self


def tune_hyper_parameters(
        args,
        opt_space,
        folds,
        info,
        feature_transform_f=None,
        feature_sampler_f=None,
        feature_opt_space=None,
):
    """Tune parameters by averaging validation scores over all CV folds.

    ``folds`` is a list of regular LAMDA ``train_val_data`` tuples.  The
    ordinary tuner is reused so that the search space and model-specific
    defaults stay identical to the non-CV path.
    """
    original_get_method = talent_utils.get_method
    folds = _CVFolds(folds)

    def cv_get_method(model_name):
        method_factory = original_get_method(model_name)
        return lambda cv_args, is_regression: _CVMethod(
            method_factory, cv_args, is_regression
        )

    talent_utils.get_method = cv_get_method
    try:
        return talent_utils.tune_hyper_parameters(
            args,
            opt_space,
            folds,
            info,
            feature_transform_f=feature_transform_f,
            feature_sampler_f=feature_sampler_f,
            feature_opt_space=feature_opt_space,
        )
    finally:
        talent_utils.get_method = original_get_method


__all__ = ['tune_hyper_parameters']
