"""Data structures."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import check_data


class ClassificationDataset:
    """Binary classification dataset.

    Attributes
    ----------
    X : 2d-array
        Observations (rows) and features (columns).
        With an addtional column of 1s for the INTERCEPT_NAME.
    y : 2d-array
        Class labels (+1, -1) with shape (n_rows, 1).
    variable_names : list of str, optional, default: None
        Names of each features. Only needed if coefficients is not passed on
        initalization. None defaults to generic variable names.
    outcome_name : str, optional, default: None
        Name of the output class.
    sample_weights : 2d array, optional, default: None
        Sample weights with shape (n_features, 1). Must all be positive.
    """
    def __init__(self, X, y, variable_names, outcome_name=None, sample_weights=None):
        self._X = X
        self._y = y
        self._variable_names = variable_names
        self._outcome_name = outcome_name
        self._sample_weights = sample_weights

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
        """Pandas dataframe."""
        return pd.DataFrame(self._X, columns=self._variable_names)

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        """Preview of dataframe."""
        with pd.option_context('display.max_rows', 6):
            with pd.option_context('display.max_columns', 6):
                return str(self.df)

    def __check_rep__(self):
        check_data(self._X, self._y, self._variable_names,
                   self._outcome_name, self._sample_weights)


@dataclass
class Bounds:
    """Data class for tracking bounds."""
    objval_min: float = 0.0
    objval_max: float = np.inf
    loss_min: float = 0.0
    loss_max: float = np.inf
    min_size: float = 0.0
    max_size: float = np.inf

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
    n_bound_updates_min_size: int = 0
    n_bound_updates_max_size: int = 0
    n_bound_updates_objval_min: int = 0
    n_bound_updates_objval_max: int = 0

    def asdict(self):
        return self.__dict__
