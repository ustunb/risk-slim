"""Data structures."""

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
    def __init__(self, X, y, variable_names, outcome_name=None, sample_weights=None, check=False):
        self._X = X
        self._y = y
        self._variable_names = variable_names
        self._outcome_name = outcome_name
        self._sample_weights = sample_weights
        if check:
            assert self.__check_rep__()

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

    def __check_rep__(self):
        return check_data(self._X, self._y, self._variable_names, self._outcome_name, self._sample_weights)

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        """Preview of dataframe."""
        with pd.option_context('display.max_rows', 6):
            with pd.option_context('display.max_columns', 6):
                return str(self.df)


