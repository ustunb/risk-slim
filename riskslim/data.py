"""Data structures."""
import warnings

import numpy as np
import pandas as pd
from riskslim.defaults import INTERCEPT_NAME
from riskslim.utils import is_integer

class ClassificationDataset:
    """Binary classification data.

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
    def __init__(self, X, y, variable_names = None, outcome_name=None, sample_weights=None):

        # todo copy
        # todo: check X, y, names
        # X should be all finite - (n_variables x n_samples matrix)
        # y should be all finite and in 0,1 or -1,1 (flat n_samples array)
        # sample_weights should be all finite, positive (flat n_samples array)
        # make sure the dimensions of X, y, and sample_weights match
        # convert y \in 0,1 to y \in -1,1 in this function?
        # output warnings if X is not binary
        # **issue warning if any column of X are constant**

        if variable_names is None:
            variable_names = default_variable_names(n_variables = X.shape[1])

        # add intercept
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        variable_names = [INTERCEPT_NAME] + variable_names

        # assign
        self._X = X
        self._y = y
        self._classes, _ = np.unique(y, return_inverse=True)
        self._variable_names = variable_names
        self._outcome_name = outcome_name
        self._sample_weights = sample_weights

        assert self.__check_rep__()

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def classes(self):
        return self._classes

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
    def d(self):
        return len(self.variable_names)

    @property
    def n(self):
        return self._X.shape[0]

    @property
    def df(self):
        """Pandas dataframe."""
        df = pd.DataFrame(self._X, columns=self._variable_names)
        df.insert(loc = 0, column = self._outcome_name, value = self._y)
        return df

    def __check_rep__(self):
        return self.check_data(self._X, self._y, self._variable_names, self._outcome_name, self._sample_weights)

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        """Preview of dataframe."""
        with pd.option_context('display.max_rows', 6):
            with pd.option_context('display.max_columns', 6):
                return str(self.df)

    @staticmethod
    def check_data(X, y, variable_names, outcome_name=None, sample_weights=None):
        """Ensures that data contains training data that is suitable for binary
        classification problems throws AssertionError if not.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the INTERCEPT_NAME.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        variable_names : list of str
            Names of each features.
        outcome_name : str, optional, default: None
            Name of the output class.
        sample_weights : 2d array.
            Sample weights with shape (n_features, 1). Must all be positive.

        Returns
        -------
        True if data passes checks
        """
        # type checks
        assert type(X) is np.ndarray, "type(X) should be numpy.ndarray"
        assert type(y) is np.ndarray, "type(y) should be numpy.ndarray"
        assert type(variable_names) is list, "variable_names should be a list"

        if outcome_name is not None:
            assert type(outcome_name) is str, "outcome_name should be a str"

        # sizes and uniqueness
        N, P = X.shape
        assert N > 0, 'X matrix must have at least 1 row'
        assert P > 0, 'X matrix must have at least 1 column'
        assert len(y) == N, 'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'
        assert len(list(set(variable_names))) == len(variable_names), 'variable_names is not unique'
        assert len(variable_names) == P, 'len(variable_names) should be same as # of cols in X'


        # feature matrix
        assert np.all(~np.isnan(X)), 'X has nan entries'
        assert np.all(~np.isinf(X)), 'X has inf entries'

        # offset in feature matrix
        if INTERCEPT_NAME in variable_names:
            assert np.all(X[:, variable_names.index(INTERCEPT_NAME)] == 1.0), "(Intercept)' column should only be composed of 1s"
        else:
            warnings.warn("there is no column named INTERCEPT_NAME in variable_names")

        # labels values
        assert np.all((y == 1) | (y == -1)), 'Need Y[i] = [-1,1] for all i.'
        if np.all(y == 1):
            warnings.warn('Y does not contain any positive examples. Need Y[i] = +1 for at least 1 i.')
        if np.all(y == -1):
            warnings.warn('Y does not contain any negative examples. Need Y[i] = -1 for at least 1 i.')

        if sample_weights is not None:
            assert type(sample_weights) is np.ndarray, 'sample_weights should be an array'
            assert len(sample_weights) == N, 'sample_weights should contain N elements'
            assert all(sample_weights > 0.0), 'sample_weights[i] > 0 for all i '

            # by default, we set sample_weights as an N x 1 array of ones. if not, then sample weights is non-trivial
            if np.any(sample_weights != 1.0) and len(np.unique(sample_weights)) < 2:
                warnings.warn('note: sample_weights only has <2 unique values')

        return True



def default_variable_names(n_variables, prefix = 'x'):
    """
    Parameters
    ----------
    n_features: # of columns in X matrix
    include_intercept = set to True to set the first variable name to INTERCEPT_NAME
    Returns
    list of unique variable names
    -------
    """
    assert is_integer(n_variables) and n_variables > 0
    assert isinstance(prefix, str) and len(prefix) > 0
    n_padding = np.floor(np.log10(n_variables + 1)).astype(int) + 1
    fmt = ':0{}d'.format(n_padding)
    namer = prefix + '{' + fmt + '}'
    names = [namer.format(j) for j in range(n_variables)]
    return names

