import logging
import sys
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import prettytable as pt
from .defaults import INTERCEPT_NAME

# DATA
def load_data_from_csv(dataset_csv_file, sample_weights_csv_file=None, fold_csv_file=None, fold_num=0):
    """Load data from a csv.

    Parameters
    ----------
    dataset_csv_file
        Csv file containing the training data
        See /datasets/adult_data.csv for an example
        Training data stored as a table with N+1 rows and d+1 columns
        Column 1 is the outcome variable entries must be (-1,1) or (0,1)
        Column 2 to d+1 are the d input variables
        Row 1 contains unique names for the outcome variable, and the input vairable
    sample_weights_csv_file
        Csv file containing sample weights for the training data
        Weights stored as a table with N rows and 1 column
        All sample weights must be non-negative
    fold_csv_file
        Csv file containing indices of folds for K-fold cross validation
        Fold indices stored as a table with N rows and 1 column
        Folds must be integers between 1 to K
        If fold_csv_file is None, then we do not use folds
    fold_num
        Int between 0 to K, where K is set by the fold_csv_file
        Let fold_idx be the N x 1 index vector listed in fold_csv_file
        Samples where fold_idx == fold_num will be used to test
        Samples where fold_idx != fold_num will be used to train the model
        Fold_num = 0 means use "all" of the training data (since all values of fold_idx \in [1,K])
        If fold_csv_file is None, then fold_num is set to 0

    Returns
    -------
    data : dict
        Cotains training data for a binary classification problem with the fields:

        - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the INTERCEPT_NAME
        - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
        - 'variable_names' list of strings containing the names of each feature (list)
        - 'Y_name' string containing the name of the output (optional)
        - 'sample_weights' N x 1 vector of sample weights, must all be positive

    """
    dataset_csv_file = Path(dataset_csv_file)
    if not dataset_csv_file.exists():
        raise IOError('could not find dataset_csv_file: %s' % dataset_csv_file)

    df = pd.read_csv(dataset_csv_file, sep = ',')

    raw_data = df.to_numpy()
    data_headers = list(df.columns.values)
    N = raw_data.shape[0]

    # Setup Y vector and Y_name
    y_col_idx = [0]
    y = raw_data[:, y_col_idx]
    y[y == 0] = -1
    outcome_name = data_headers[y_col_idx[0]]

    # Setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j not in y_col_idx]
    X = raw_data[:, X_col_idx]
    variable_names = [data_headers[j] for j in X_col_idx]

    # Insert a column of ones to X for the intercept
    X = np.insert(arr=X, obj=0, values=np.ones(N), axis=1)
    variable_names.insert(0, INTERCEPT_NAME)

    if sample_weights_csv_file is None:
        sample_weights = np.ones(N)
    else:
        sample_weights_csv_file = Path(sample_weights_csv_file)
        if not sample_weights_csv_file.exists():
            raise IOError('could not find sample_weights_csv_file: %s' % sample_weights_csv_file)
        sample_weights = pd.read_csv(sample_weights_csv_file, sep=',', header=None)
        sample_weights = sample_weights.to_numpy()

    data = {
        'X': X,
        'y': y,
        'variable_names': variable_names,
        'outcome_name': outcome_name,
        'sample_weights': sample_weights,
    }

    # Load folds
    if fold_csv_file is not None:
        fold_csv_file = Path(fold_csv_file)
        if not fold_csv_file.exists():
            raise IOError('could not find fold_csv_file: %s' % fold_csv_file)

        fold_idx = pd.read_csv(fold_csv_file, sep=',', header=None)
        fold_idx = fold_idx.values.flatten()
        K = max(fold_idx)
        all_fold_nums = np.sort(np.unique(fold_idx))
        assert len(fold_idx) == N, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), N)
        assert np.all(all_fold_nums == np.arange(1, K+1)), "folds should contain indices between 1 to %r" % K
        assert fold_num in np.arange(0, K+1), "fold_num should either be 0 or an integer between 1 to %r" % K
        if fold_num >= 1:
            train_idx = fold_num != fold_idx
            data['X'] = data['X'][train_idx,]
            data['y'] = data['y'][train_idx]
            data['sample_weights'] = data['sample_weights'][train_idx]

    assert check_data(X, y, variable_names, outcome_name, sample_weights)

    return data


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


# MODEL PRINTING
def print_model(rho, variable_names, outcome_name, show_omitted_variables=False, return_only=False):

    rho_values = np.copy(rho)
    rho_names = list(variable_names)

    if INTERCEPT_NAME in rho_names:
        intercept_ind = variable_names.index(INTERCEPT_NAME)
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.remove(INTERCEPT_NAME)
    else:
        intercept_val = 0

    if outcome_name is None:
        predict_string = "Pr(Y = +1) = 1.0/(1.0 + exp(-(%d + score))" % intercept_val
    else:
        predict_string = "Pr(%s = +1) = 1.0/(1.0 + exp(-(%d + score))" % (outcome_name.upper(), intercept_val)

    if not show_omitted_variables:
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]

        #sort by most positive to most negative
        sort_ind = np.argsort(-np.array(rho_values))
        rho_values = [rho_values[j] for j in sort_ind]
        rho_names = [rho_names[j] for j in sort_ind]
        rho_values = np.array(rho_values)

    rho_values_string = [str(int(i)) + " points" for i in rho_values]
    n_variable_rows = len(rho_values)
    total_string = "ADD POINTS FROM ROWS %d to %d" % (1, n_variable_rows)

    max_name_col_length = max(len(predict_string), len(total_string), max([len(s) for s in rho_names])) + 2
    max_value_col_length = max(7, max([len(s) for s in rho_values_string]) + len("points")) + 2

    m = pt.PrettyTable()
    m.field_names = ["Variable", "Points", "Tally"]

    m.add_row([predict_string, "", ""])
    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])

    for name, value_string in zip(rho_names, rho_values_string):
        m.add_row([name, value_string, "+ ....."])

    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])
    m.add_row([total_string, "SCORE", "= ....."])
    m.header = False
    m.align["Variable"] = "l"
    m.align["Points"] = "r"
    m.align["Tally"] = "r"

    if not return_only:
        print(m)

    return m


# LOGGING
def setup_logging(logger, log_to_console = True, log_file = None):
    """
    Sets up logging to console and file on disk
    See https://docs.python.org/2/howto/logging-cookbook.html for details on how to customize

    Parameters
    ----------
    log_to_console  set to True to disable logging in console
    log_file        path to file for loggin

    Returns
    -------
    Logger object that prints formatted messages to log_file and console
    """

    # quick return if no logging to console or file
    if log_to_console is False and log_file is None:
        logger.disabled = True
        return logger

    log_format = logging.Formatter(fmt='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p')

    # log to file
    if log_file is not None:
        fh = logging.FileHandler(filename=log_file)
        #fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        #ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_format)
        logger.addHandler(ch)

    return logger


def print_log(msg, print_flag = True):
    """

    Parameters
    ----------
    msg
    print_flag

    Returns
    -------

    """
    if print_flag:
        if isinstance(msg, str):
            print('%s | %s' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg))
        else:
            print('%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg))
        sys.stdout.flush()

# Settings
def validate_settings(settings = None, defaults = None):

    if settings is None:
        settings = dict()
    else:
        assert isinstance(settings, dict)
        settings = dict(settings)

    if defaults is not None:
        assert isinstance(defaults, dict)
        settings = {k: settings[k] if k in settings else defaults[k] for k in defaults}

    return settings

# Data Types
def is_integer(x):
    """
    checks if numpy array is an integer vector

    Parameters
    ----------
    x

    Returns
    -------

    """
    return np.array_equal(x, np.require(x, dtype=np.int_))


def cast_to_integer(x):
    """
    casts numpy array to integer vector

    Parameters
    ----------
    x

    Returns
    -------

    """
    original_type = x.dtype
    return np.require(np.require(x, dtype=np.int_), dtype=original_type)