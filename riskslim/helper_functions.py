import sys
import time
import numpy as np
import pandas as pd
import os.path
import warnings

# PRINTING AND LOGGING
def print_log(msg, print_flag = True):
    if print_flag:
        if type(msg) is str:
            print ('%s | ' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()))) + msg
        else:
            print '%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg)
        sys.stdout.flush()

def get_rho_string(rho, vtypes='I'):
    if len(vtypes) == 1:
        if vtypes == 'I':
            rho_string = ' '.join(map(lambda x: str(int(x)), rho))
        else:
            rho_string = ' '.join(map(lambda x: str(x), rho))
    else:
        rho_string = ''
        for j in range(0, len(rho)):
            if vtypes[j] == 'I':
                rho_string += ' ' + str(int(rho[j]))
            else:
                rho_string += (' %1.6f' % rho[j])

    return rho_string

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

def get_or_set_default(settings, setting_name, default_value, type_check=False, print_flag=True):
    if setting_name in settings:
        if type_check:
            # check type match
            default_type = type(default_value)
            user_type = type(settings[setting_name])
            if user_type == default_type:
                settings[setting_name] = default_value
            else:
                print_log("type mismatch on %s: user provided type: %s and but expected type: %s" % (
                setting_name, user_type, default_type), print_flag)
                print_log("setting %s to its default value: %r" % (setting_name, default_value), print_flag)
                settings[setting_name] = default_value
                # else: do nothing
    else:
        print_log("setting %s to its default value: %r" % (setting_name, default_value), print_flag)
        settings[setting_name] = default_value

    return settings

# LOADING DATA FROM DISK
def load_data_from_csv(dataset_csv_file, sample_weights_csv_file = None):

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
    Y_name = [data_headers[j] for j in Y_col_idx]
    Y[Y == 0] = -1

    # setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j not in Y_col_idx]
    X = raw_data[:, X_col_idx]
    X_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    X = np.insert(arr=X, obj=0, values=np.ones(N), axis=1)
    X_names.insert(0, '(Intercept)')

    if sample_weights_csv_file is None:
        sample_weights = np.ones(N)
    else:
        if os.path.isfile(sample_weights_csv_file):
            sample_weights = pd.read_csv(sample_weights_csv_file, sep=',')
            sample_weights = sample_weights.as_matrix()
        else:
            raise IOError('could not find sample_weights_csv_file: %s' % sample_weights_csv_file)

    data = {
        'X': X,
        'X_names': X_names,
        'Y': Y,
        'Y_name': Y_name,
        'sample_weights': sample_weights,
    }


    return data

# DATA CHECK
def check_data(X, X_names, Y):
    # type checks
    assert type(X) is np.ndarray, "type(X) should be numpy.ndarray"
    assert type(Y) is np.ndarray, "type(Y) should be numpy.ndarray"
    assert type(X_names) is list, "X_names should be a list"

    # sizes and uniqueness
    N, P = X.shape
    assert N > 0, 'X matrix must have at least 1 row'
    assert P > 0, 'X matrix must have at least 1 column'
    assert len(Y) == N, 'len(Y) should be same as # of rows in X'
    assert len(list(set(X_names))) == len(X_names), 'X_names is not unique'
    assert len(X_names) == P, 'len(X_names) should be same as # of cols in X'

    # X_matrix values
    if '(Intercept)' in X_names:
        assert all(X[:, X_names.index('(Intercept)')] == 1.0), "'(Intercept)' column should only be composed of 1s"
    else:
        warnings.warn("there is no column named '(Intercept)' in X_names")
    assert np.all(~np.isnan(X)), 'X has nan entries'
    assert np.all(~np.isinf(X)), 'X has inf entries'

    # Y vector values
    assert all((Y == 1) | (Y == -1)), 'Y[i] should = [-1,1] for all i'
    if all(Y == 1):
        warnings.warn("all Y_i == 1 for all i")
    if all(Y == -1):
        warnings.warn("all Y_i == -1 for all i")

        # TODO (optional) collect warnings and return those?

