"""Cross-validation hyper-parameter tuning for LAMDA-TALENT datasets."""

from __future__ import annotations

import copy
import sys

import numpy as np

from catkernel.talent import utils as talent_utils


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
        for fold_data in folds:
            method = self._method_factory(self._args, self._is_regression)
            method.fit(fold_data, info, train=train, config=copy.deepcopy(config))
            score = method.trlog.get('best_res')
            assert score is not None
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
                self.trlog[key] = sum(values) / len(values)

        sys.stderr.write(f"trial: {fold_scores}\n")
        self.trlog['fold_scores'] = fold_scores
        return self


def tune_hyper_parameters(args, opt_space, folds, info):
    """Tune parameters by averaging validation scores over all CV folds.

    ``folds`` is a list of regular LAMDA ``train_val_data`` tuples.  The
    ordinary tuner is reused so that the search space and model-specific
    defaults stay identical to the non-CV path.
    """
    original_get_method = talent_utils.get_method

    def cv_get_method(model_name):
        method_factory = original_get_method(model_name)
        return lambda cv_args, is_regression: _CVMethod(
            method_factory, cv_args, is_regression
        )

    talent_utils.get_method = cv_get_method
    try:
        return talent_utils.tune_hyper_parameters(args, opt_space, folds, info)
    finally:
        talent_utils.get_method = original_get_method


__all__ = ['tune_hyper_parameters']
