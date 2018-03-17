import sys
import time
import os.path
import warnings
import numpy as np
import pandas as pd
import prettytable as pt
import logging

# PRINTING AND LOGGING
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
        if type(msg) is str:
            print ('%s | ' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()))) + msg
        else:
            print '%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg)
        sys.stdout.flush()


def get_rho_string(rho, vtypes = 'I'):

    if len(vtypes) == 1:
        if vtypes == 'I':
            rho_string = ' '.join(map(lambda x: str(int(x)), rho))
        else:
            rho_string = ' '.join(map(lambda x: str(x), rho))

    else:
        rho_string = ''
        for j in range(len(rho)):
            if vtypes[j] == 'I':
                rho_string += ' ' + str(int(rho[j]))
            else:
                rho_string += (' %1.6f' % rho[j])

    return rho_string


def print_update(rho, n_iterations, upperbound, lowerbound, relative_gap, vtypes = 'C'):
    rho_string = get_rho_string(rho, vtypes)
    print_log("cuts = %d \t UB = %.5f \t LB = %.5f \t GAP = %.5f%%" % (n_iterations, upperbound, lowerbound, 100.0 * relative_gap))
    print_log('rho:%s\n' % rho_string)


def easy_type(data_value):
    type_name = type(data_value).__name__
    if type_name in {"list", "set"}:
        types = {easy_type(item) for item in data_value}
        if len(types) == 1:
            return next(iter(types))
        elif types.issubset({"int", "float"}):
            return "float"
        else:
            return "multiple"
    elif type_name == "str":
        if data_value in {'True', 'TRUE'}:
            return "bool"
        elif data_value in {'False', 'FALSE'}:
            return "bool"
        else:
            return "str"
    elif type_name == "int":
        return "int"
    elif type_name == "float":
        return "float"
    elif type_name == "bool":
        return "bool"
    else:
        return "unknown"


def convert_str_to_bool(val):
    val = val.lower().strip()
    if val == 'true':
        return True
    elif val == 'false':
        return False
    else:
        return None


def get_or_set_default(settings, setting_name, default_value, type_check = False, print_flag=True):

    if setting_name not in settings:

        print_log("setting %s to its default value: %r" %
                  (setting_name, default_value), print_flag)

        settings[setting_name] = default_value

    elif setting_name in settings and type_check:

        default_type = type(default_value)
        user_type = type(settings[setting_name])

        if user_type is not default_type:
            print_log("type mismatch on %s:\nuser provided %s\n expected %s" %
                      (setting_name, user_type, default_type), print_flag)

            print_log("setting %s to its default value: %r" %
                      (setting_name, default_value), print_flag)

            settings[setting_name] = default_value

    return settings


# LOADING DATA FROM DISK
def load_data_from_csv(dataset_csv_file, sample_weights_csv_file = None, fold_csv_file = None, fold_num = 0):
    """

    Parameters
    ----------
    dataset_csv_file                csv file containing the training data
                                    see /datasets/adult_data.csv for an example
                                    training data stored as a table with N+1 rows and d+1 columns
                                    column 1 is the outcome variable entries must be (-1,1) or (0,1)
                                    column 2 to d+1 are the d input variables
                                    row 1 contains unique names for the outcome variable, and the input vairable

    sample_weights_csv_file         csv file containing sample weights for the training data
                                    weights stored as a table with N rows and 1 column
                                    all sample weights must be non-negative

    fold_csv_file                   csv file containing indices of folds for K-fold cross validation
                                    fold indices stored as a table with N rows and 1 column
                                    folds must be integers between 1 to K
                                    if fold_csv_file is None, then we do not use folds

    fold_num                        int between 0 to K, where K is set by the fold_csv_file
                                    let fold_idx be the N x 1 index vector listed in fold_csv_file
                                    samples where fold_idx == fold_num will be used to test
                                    samples where fold_idx != fold_num will be used to train the model
                                    fold_num = 0 means use "all" of the training data (since all values of fold_idx \in [1,K])
                                    if fold_csv_file is None, then fold_num is set to 0


    Returns
    -------
    dictionary containing training data for a binary classification problem with the fields:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)
     - 'Y_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    """

    if os.path.isfile(dataset_csv_file):
        df = pd.read_csv(dataset_csv_file, sep=',')
    else:
        raise IOError('could not find dataset_csv_file: %s' % dataset_csv_file)

    raw_data = df.as_matrix()
    data_headers = list(df.columns.values)
    N = raw_data.shape[0]

    # setup Y vector and Y_name
    Y_col_idx = [0]
    Y = raw_data[:, Y_col_idx]
    Y_name = data_headers[Y_col_idx[0]]
    Y[Y == 0] = -1

    # setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j not in Y_col_idx]
    X = raw_data[:, X_col_idx]
    variable_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    X = np.insert(arr=X, obj=0, values=np.ones(N), axis=1)
    variable_names.insert(0, '(Intercept)')

    if sample_weights_csv_file is None:
        sample_weights = np.ones(N)
    else:
        if os.path.isfile(sample_weights_csv_file):
            sample_weights = pd.read_csv(sample_weights_csv_file, sep=',', header=None)
            sample_weights = sample_weights.as_matrix()
        else:
            raise IOError('could not find sample_weights_csv_file: %s' % sample_weights_csv_file)

    data = {
        'X': X,
        'Y': Y,
        'variable_names': variable_names,
        'outcome_name': Y_name,
        'sample_weights': sample_weights,
        }

    #load folds
    if fold_csv_file is not None:
        if not os.path.isfile(fold_csv_file):
            raise IOError('could not find fold_csv_file: %s' % fold_csv_file)
        else:
            fold_idx = pd.read_csv(fold_csv_file, sep=',', header=None)
            fold_idx = fold_idx.values.flatten()
            K = max(fold_idx)
            all_fold_nums = np.sort(np.unique(fold_idx))
            assert len(fold_idx) == N, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), N)
            assert np.all(all_fold_nums == np.arange(1, K+1)), "folds should contain indices between 1 to %r" % K
            assert fold_num in np.arange(0, K+1), "fold_num should either be 0 or an integer between 1 to %r" % K
            if fold_num >= 1:
                test_idx = fold_num == fold_idx
                train_idx = fold_num != fold_idx
                data['X'] = data['X'][train_idx,]
                data['Y'] = data['Y'][train_idx]
                data['sample_weights'] = data['sample_weights'][train_idx]

    assert check_data(data)
    return data


# DATA CHECK
def check_data(data):
    """
    makes sure that 'data' contains training data that is suitable for binary classification problems
    throws AssertionError if

    'data' is a dictionary that must contain:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)

     data can also contain:

     - 'outcome_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    Returns
    -------
    True if data passes checks

    """
    # type checks
    assert type(data) is dict, "data should be a dict"

    assert 'X' in data, "data should contain X matrix"
    assert type(data['X']) is np.ndarray, "type(X) should be numpy.ndarray"

    assert 'Y' in data, "data should contain Y matrix"
    assert type(data['Y']) is np.ndarray, "type(Y) should be numpy.ndarray"

    assert 'variable_names' in data, "data should contain variable_names"
    assert type(data['variable_names']) is list, "variable_names should be a list"

    X = data['X']
    Y = data['Y']
    variable_names = data['variable_names']

    if 'outcome_name' in data:
        assert type(data['outcome_name']) is str, "outcome_name should be a str"

    # sizes and uniqueness
    N, P = X.shape
    assert N > 0, 'X matrix must have at least 1 row'
    assert P > 0, 'X matrix must have at least 1 column'
    assert len(Y) == N, 'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'
    assert len(list(set(data['variable_names']))) == len(data['variable_names']), 'variable_names is not unique'
    assert len(data['variable_names']) == P, 'len(variable_names) should be same as # of cols in X'

    # feature matrix
    assert np.all(~np.isnan(X)), 'X has nan entries'
    assert np.all(~np.isinf(X)), 'X has inf entries'

    # offset in feature matrix
    if '(Intercept)' in variable_names:
        assert all(X[:, variable_names.index('(Intercept)')] == 1.0), "(Intercept)' column should only be composed of 1s"
    else:
        warnings.warn("there is no column named '(Intercept)' in variable_names")

    # labels values
    assert all((Y == 1) | (Y == -1)), 'Need Y[i] = [-1,1] for all i.'
    if all(Y == 1):
        warnings.warn('Y does not contain any positive examples. Need Y[i] = +1 for at least 1 i.')
    if all(Y == -1):
        warnings.warn('Y does not contain any negative examples. Need Y[i] = -1 for at least 1 i.')

    if 'sample_weights' in data:
        sample_weights = data['sample_weights']
        type(sample_weights) is np.ndarray
        assert len(sample_weights) == N, 'sample_weights should contain N elements'
        assert all(sample_weights > 0.0), 'sample_weights[i] > 0 for all i '

        # by default, we set sample_weights as an N x 1 array of ones. if not, then sample weights is non-trivial
        if any(sample_weights != 1.0) and len(np.unique(sample_weights)) < 2:
            warnings.warn('note: sample_weights only has <2 unique values')

    return True


def print_model(rho, data,  show_omitted_variables = False):

    variable_names = data['variable_names']

    rho_values = np.copy(rho)
    rho_names = list(variable_names)

    if '(Intercept)' in rho_names:
        intercept_ind = variable_names.index('(Intercept)')
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.remove('(Intercept)')
    else:
        intercept_val = 0

    if 'outcome_name' in data:
        predict_string = "Pr(Y = +1) = 1/(1 + exp(%d - score))" % intercept_val
    else:
        predict_string = "Pr(%s = +1) = 1/(1 + exp(%d - score)" % (data['outcome_name'].upper(), intercept_val)

    if not show_omitted_variables:
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]
        rho_binary = [np.all((data['X'][:,j] == 0) | (data['X'][:,j] == 1)) for j in selected_ind]

        #sort by most positive to most negative
        sort_ind = np.argsort(-np.array(rho_values))
        rho_values = [rho_values[j] for j in sort_ind]
        rho_names = [rho_names[j] for j in sort_ind]
        rho_binary = [rho_binary[j] for j in sort_ind]
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

    print(m)
    return m


# DATA CONVERSION
def is_integer(rho):
    """
    checks if numpy array is an integer vector

    Parameters
    ----------
    rho

    Returns
    -------

    """
    return np.array_equal(rho, np.require(rho, dtype=np.int_))


def cast_to_integer(rho):
    """
    casts numpy array to integer vector
    Parameters
    ----------
    rho

    Returns
    -------

    """
    original_type = rho.dtype
    return np.require(np.require(rho, dtype=np.int_), dtype=original_type)



