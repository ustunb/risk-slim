"""Data structures."""

import warnings
from dataclasses import dataclass
import numpy as np


class ClassificationDataset(object):

    def __init__(self, **kwargs):
        raise NotImplementedError()

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def variable_names(self):
        return self._variable_names

    @property
    def outcome_name(self):
        return self._outcome_name

    @property
    def df(self):
        """
        :return:
        """
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __check_rep__(self):

        X = self._X
        Y = self._y
        variable_names = self._variable_names
        outcome_name = self._outcome_name

        # sizes and uniqueness
        N, P = X.shape
        assert N > 0, 'X matrix must have at least 1 row'
        assert P > 0, 'X matrix must have at least 1 column'
        assert len(Y) == N, 'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'
        assert len(list(variable_names)) == len(variable_names), 'variable_names is not unique'
        assert len(variable_names) == P, 'len(variable_names) should be same as # of cols in X'

        # feature matrix
        assert np.isfinite(X).all(), 'X must have finite entries'

        # offset in feature matrix
        if '(Intercept)' in variable_names:
            assert all(X[:, variable_names.index('(Intercept)')] == 1.0), "'(Intercept)' feature should be a column of 1s"
        else:
            warnings.warn("no column named '(Intercept)' in variable_names")

        # labels values
        assert np.isin(Y, [-1, 1]).all(), 'Need Y[i] = [-1,1] for all i.'

        if all(Y == 1):
            warnings.warn('Y does not contain any positive examples. Need Y[i] = +1 for at least 1 i.')
        elif all(Y == -1):
            warnings.warn('Y does not contain any negative examples. Need Y[i] = -1 for at least 1 i.')

        # sample weights
        sample_weights = self._sample_weights
        assert isinstance(sample_weights, np.ndarray)
        assert len(sample_weights) == N, 'sample_weights should contain N elements'
        assert all(sample_weights > 0.0), 'sample_weights[i] > 0 for all i '

@dataclass
class Bounds:
    """Data class for tracking bounds."""
    objval_min: float = 0.0
    objval_max: float = np.inf
    loss_min: float = 0.0
    loss_max: float = np.inf
    L0_min: float = 0.0
    L0_max: float = np.inf

    def asdict(self):
        return self.__dict__

@dataclass
class Stats:
    """Data class for tracking statistics."""
    incumbent: np.ndarray
    upperbound: float = np.inf
    bounds: Bounds = Bounds()
    lowerbound: float = 0.0
    relative_gap: float = np.inf
    nodes_processed: int = 0
    nodes_remaining: int = 0
    # Time
    start_time: float = np.nan
    total_run_time: float = 0.0
    total_cut_time: float = 0.0
    total_polish_time: float = 0.0
    total_round_time: float = 0.0
    total_round_then_polish_time: float = 0.0
    # Cuts
    cut_callback_times_called: int = 0
    heuristic_callback_times_called: int = 0
    total_cut_callback_time: float = 0.0
    total_heuristic_callback_time: float = 0.0
    # Number of times solutions were updates
    n_incumbent_updates: int = 0
    n_heuristic_updates: int = 0
    n_cuts: int = 0
    n_polished: int = 0
    n_rounded: int = 0
    n_rounded_then_polished: int = 0
    # Total # of bound updates
    n_update_bounds_calls: int = 0
    n_bound_updates: int = 0
    n_bound_updates_loss_min: int = 0
    n_bound_updates_loss_max: int = 0
    n_bound_updates_L0_min: int = 0
    n_bound_updates_L0_max: int = 0
    n_bound_updates_objval_min: int = 0
    n_bound_updates_objval_max: int = 0

    def asdict(self):
        return self.__dict__
