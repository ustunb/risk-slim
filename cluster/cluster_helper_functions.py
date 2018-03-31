import os
import numpy as np
import scipy.io as sio
from riskslim.helper_functions import print_log, convert_str_to_bool

# LOADING SETTINGS FROM DISK

def load_or_set_default(setting_name, default_value, setting_dir = None, setting_header = None, print_flag = False):

    if setting_dir is None:
        if "run_dir" in vars():
            setting_dir = run_dir
        else:
            setting_dir = os.getcwd()+"/"

    if setting_header is None:
        if "run_header" in vars():
            setting_header = run_header
        else:
            setting_header = ""

    setting_type = easy_type(default_value)
    setting_file_short = setting_header + "_" + setting_name + ".setting"
    setting_file = setting_dir + setting_file_short
    # default_value = True
    # setting_dir = '/Users/berk/Desktop/Dropbox (MIT)/Research/SLIM/Run/breastcancer_F_K03N01/'
    # setting_file = 'breastcancer_F_K03N01_I_NONE_fold_3_W_NONE_L_U000_X_TEST_pos_1.000000000_only_use_best_solution.setting'
    # setting_file = setting_dir + setting_file
    if print_flag:
        print_log("setting name: %r" % setting_name)
        print_log("default value: %r" % default_value)
        print_log("default type: %s" % setting_type)
        print_log("setting dir: %r" % setting_dir)
        print_log("setting file: %r" % (setting_header + "_" + setting_name))

    # load setting value
    if os.path.isfile(setting_file):

        if print_flag:
            print_log("reading setting_file on disk")

        if setting_type is "bool":
            setting_value = np.loadtxt(fname=setting_file, dtype="str")
            if setting_value.size > 1:
                setting_value = setting_value.tolist()
                setting_value = [convert_str_to_bool(x) for x in setting_value]
            else:
                setting_value = setting_value.tolist()
                setting_value = convert_str_to_bool(setting_value)

        elif setting_type in {"int", "float", "str"}:
            setting_value = np.loadtxt(fname=setting_file, dtype="str")
            if setting_type in {"int", "float"}:
                if np.all(setting_value == 'inf'):
                    setting_value = np.array(float('inf'))
                else:
                    setting_value = np.require(setting_value, dtype = setting_type)

            setting_value = setting_value.tolist()

        else:
            try:
                setting_value = np.loadtxt(fname=setting_file)
            except:
                try:
                    setting_value = np.loadtxt(fname=setting_file, dtype='str')
                except:
                    print_log("using default value (failed to load data from disk)")
                    setting_value = default_value

            try:
                setting_value = setting_value.tolist()
            except:
                pass
    else:
        if print_flag:
            print_log("using default value (could not find setting file on disk)")

        setting_value = default_value

    if print_flag:
        print_log("final value: %r\n" % setting_value)

    return setting_value

def get_default_if_type_mismatch(setting_value, default_value):
    if type(setting_value) == type(default_value):
        return setting_value
    else:
        return default_value

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


#LOADING DATASETS

def convert_matlab_lset(rawlset):

    variable_names = [str(x) for x in rawlset['name'].tolist()]
    lim = rawlset['lim'].tolist()
    lb = [x[0] for x in lim]
    ub = [x[1] for x in lim]
    C_0j = rawlset['C_0j'].astype('float').tolist()
    sign = rawlset['sign'].astype('float')
    sign[np.isnan(sign)] = 0
    sign = sign.astype('int').tolist()

    myLset = Lset(variable_names = variable_names,
                  lb = lb,
                  ub = ub,
                  C_0j = C_0j,
                  sign = sign)

    return myLset

def get_from_hset(hset, hid, fname, default_value):
    if fname in hset.dtype.names:
        if hset[fname].ndim == 0:
            return hset[fname]
        else:
            return hset[fname][hid]
    else:
        return default_value

def create_custom_coefficient_set(Z_min, Z_max, variable_names, max_offset, max_coefficient, max_L0_value):

    max_coefficient = abs(max_coefficient)
    custom_Lset = Lset(variable_names = variable_names, lb = -max_coefficient, ub = max_coefficient, sign = 0)
    if '(Intercept)' in variable_names:
        P = len(Z_min)
        intercept_ind = custom_Lset.variable_names.index('(Intercept)')
        variable_ind = [i for i in range(0, P) if not i == intercept_ind]
        L0_max = min(max_L0_value, P - 1) if max_L0_value > 0 else P - 1

        if max_offset == -1:
            s_min, s_max = get_score_bounds(Z_min = Z_min[variable_ind],
                                            Z_max = Z_max[variable_ind],
                                            rho_lb = custom_Lset.lb[variable_ind],
                                            rho_ub = custom_Lset.ub[variable_ind],
                                            L0_reg_ind = np.isnan(custom_Lset.C_0j)[variable_ind],
                                            L0_max = L0_max)

            max_offset = max(abs(s_min), abs(s_max))

        custom_Lset.set_field('lb', '(Intercept)', -max_offset)
        custom_Lset.set_field('ub', '(Intercept)', max_offset)

    return custom_Lset

def load_matlab_data(data_file_name, fold_id, fold_num = 0, inner_fold_id = 'NONE', sample_weight_id = 'NONE'):
    
    raw_data = sio.loadmat(file_name = data_file_name,
                           matlab_compatible = False,
                           chars_as_strings = True,
                           squeeze_me = True,
                           variable_names = ['X', 'Y', 'cvindices', 'X_headers', 'Y_headers', 'X_test', 'Y_test', 'sample_weights', 'sample_weights_test'],
                           verify_compressed_data_integrity = False)

    X = raw_data['X'].astype('float')
    N, P = X.shape
    Y = raw_data['Y'].astype('float').reshape(N, 1)

    #load test data
    found_test_data_in_mat_file = ('X_test' in raw_data and
                                   'Y_test' in raw_data and
                                   len(raw_data['X_test']) > 0 and
                                   len(raw_data['Y_test']) > 0)

    if found_test_data_in_mat_file:
        X_test = raw_data['X_test'].astype('float')
        Y_test = raw_data['Y_test'].astype('float').reshape(X_test.shape[0], 1)
        N_test = X_test.shape[0]
    else:
        X_test = np.zeros(shape = (0, P), dtype = X.dtype)
        Y_test = np.zeros(shape = (0, 1), dtype = Y.dtype)
        N_test = 0

    #load folds
    K_outer_id = fold_id[0:3]
    N_outer_id = fold_id[3:6]
    K_outer = int(K_outer_id[1:])
    N_outer = int(N_outer_id[1:])

    folds = raw_data['cvindices'][K_outer_id].item(0)
    if (folds.ndim == 1) & (folds.shape[0] == N):
        folds = folds
    else:
        folds = folds[:, N_outer - 1]

    #load sample weights
    if found_test_data_in_mat_file:
        found_sample_weights_in_mat_file = ('sample_weights' in raw_data and 'sample_weights_test' in raw_data)
    else:
        found_sample_weights_in_mat_file = ('sample_weights' in raw_data)

    sample_weight_flag = sample_weight_id != "NONE"

    if found_sample_weights_in_mat_file and sample_weight_flag:
        sid = int(sample_weight_id[1:]) - 1
        print_log("LOADING SAMPLE WEIGHTS")
        sample_weights = raw_data['sample_weights'].astype('float')
        sample_weights = sample_weights[:, sid]
        if N_test > 0:
            sample_weights_test = raw_data['sample_weights_test'].astype('float')
            sample_weights_test = sample_weights_test[:, sid]
        else:
            sample_weights_test = np.zeros(shape = Y_test.shape, dtype = sample_weights.dtype)
    else:
        sample_weights = np.ones(shape = Y.shape, dtype = 'float')
        sample_weights_test = np.ones(shape = Y_test.shape, dtype = 'float')

    sample_weights = sample_weights.astype('float').reshape(Y.shape)
    sample_weights_test = sample_weights_test.astype('float').reshape(Y_test.shape)
    #inner_fold_id format is: 'F%02dK02d' % (F_inner, K_inner)
    #F_inner = inner fold to define test set + full training set
    #K_inner = # of folds in internal cross validation
    #F < K_outer

    #check that inner CV is valid
    inner_cv_flag = inner_fold_id != "NONE"

    if inner_cv_flag:
        F_inner_id = inner_fold_id[0:3]
        F_inner = int(F_inner_id[1:])
        K_inner_id = inner_fold_id[3:6]
        K_inner = int(K_inner_id[1:])

        assert F_inner > 0, 'inner fold number must be > 0; parsed F_inner as %r' % F_inner_id
        assert F_inner <= K_outer, 'inner fold number must be < K; parsed F_inner as %r' % F_inner_id
        assert K_inner > 1, 'inner CV # of folds must be > 1; parsed as %s' % K_inner_id

        inner_cv_key = '%s_%s' % (fold_id, inner_fold_id)
        inner_folds = raw_data['cvindices'][inner_cv_key].item(0)

        #create new test set and folds
        test_ind = folds == F_inner
        keep_ind = ~test_ind
        assert len(inner_folds) == sum(keep_ind), 'size of inner fold indices does not match'

        print_log("LOADING INNER CV INDICES")
        print_log("outer fold_id: %r" % fold_id)
        print_log("inner fold_id: %r" % inner_fold_id)
        print_log("outer fold #: %r of %r" % (F_inner, K_outer))
        print_log("inner fold #: %r of %r" % (fold_num, K_inner))

        X_test = X[test_ind, :]
        Y_test = Y[test_ind, :]
        sample_weights_test = sample_weights[test_ind, :]

        X = X[keep_ind, :]
        Y = Y[keep_ind, :]
        sample_weights = sample_weights[keep_ind, :]

        folds = inner_folds
        N = sum(keep_ind)
        N_test = sum(test_ind)

    #sanity sizing checks
    assert X.shape[0] == N, 'X has incorrect shape'
    assert X.shape[1] == P, 'X has incorrect shape'
    assert Y.shape[0] == N, 'Y has incorrect shape'
    assert Y.shape[1] == 1, 'Y has incorrect shape'
    assert len(folds) == N, 'folds has incorrect shape'
    assert X_test.shape[0] == N_test, 'X_test has incorrect shape'
    assert Y_test.shape[0] == N_test, 'Y_test has incorrect shape'
    assert X_test.shape[1] == P, 'X_test has incorrect shape'
    assert Y_test.shape[1] == 1, 'Y_test has incorrect shape'
    assert sample_weights.shape[0] == N, 'sample_weights has incorrect shape'
    assert sample_weights.shape[1] == 1, 'sample_weights has incorrect shape'
    assert sample_weights_test.shape[0] == N_test, 'sample_weights_test has incorrect shape'
    assert sample_weights_test.shape[1] == 1, 'sample_weights has_test incorrect shape'

    #split data into validation and training
    valid_ind = folds == fold_num
    train_ind = ~valid_ind

    data = {
        'X': X[train_ind, :],
        'Y': Y[train_ind, :],
        'X_valid': X[valid_ind, :],
        'Y_valid': Y[valid_ind, :],
        'X_test': X_test,
        'Y_test': Y_test,
        'folds': folds,
        'sample_weights': sample_weights[train_ind, :],
        'sample_weights_valid': sample_weights[valid_ind, :],
        'sample_weights_test': sample_weights_test,
        'X_headers': [str(x) for x in raw_data['X_headers'].tolist()],
        'Y_headers': str(raw_data['Y_headers'])
    }

    return data

def load_hard_constraints(data, data_file_name, hcon_id = 'U000', use_custom_coefficient_set = False, max_coefficient = 10, max_offset = -1, max_L0_value = -1):
    
    #load data
    P = data['X'].shape[1]
    hard_constraints = {}
    
    #load coefficient set
    if use_custom_coefficient_set:
        
        Z = data['X'] * data['Y']
        hard_constraints['coef_set'] = create_custom_coefficient_set(Z_min = np.min(Z, axis = 0),
                                                                     Z_max = np.max(Z, axis = 0),
                                                                     variable_names = data['X_headers'],
                                                                     max_offset = max_offset,
                                                                     max_coefficient = max_coefficient,
                                                                     max_L0_value = max_L0_value)
        
        L0_max_trivial = P - np.sum(hard_constraints['coef_set'].C_0j == 0)
        hard_constraints['hcon_id'] = 'U000'
        hard_constraints['L0_min'] = 0
        hard_constraints['L0_max'] = min(L0_max_trivial, max_L0_value) if max_L0_value > 0 else L0_max_trivial
        hard_constraints['max_offset'] = max_offset
        hard_constraints['max_coefficient'] = max_coefficient
        hard_constraints['max_L0_value'] = max_L0_value
    
    else:
        
        mat = sio.loadmat(file_name = data_file_name,
                          matlab_compatible = False,
                          chars_as_strings = True,
                          squeeze_me = True,
                          variable_names = ['HardConstraints'],
                          verify_compressed_data_integrity = False)
        
        hid = int(hcon_id[1:]) - 1
        if 'Lset' in mat['HardConstraints'].dtype.names:
            if mat['HardConstraints']['Lset'].ndim == 0:
                hard_constraints['coef_set'] = convert_matlab_lset(mat['HardConstraints']['Lset'].item(0))
            else:
                hard_constraints['coef_set'] = convert_matlab_lset(mat['HardConstraints']['Lset'][hid])
        else:
            hard_constraints['coef_set'] = Lset(variable_names = data['X_headers'])
        
        L0_max_trivial = P - np.sum(hard_constraints['coef_set'].C_0j == 0)
        
        hard_constraints['hcon_id'] = hcon_id
        hard_constraints['L0_min'] = max(0, get_from_hset(mat['HardConstraints'], hid, 'L0_min', 0))
        hard_constraints['L0_max'] = min(L0_max_trivial, get_from_hset(mat['HardConstraints'], hid, 'L0_max', L0_max_trivial))
        
        #set max_offset, max_L0_value, max_coefficient using actual values
        rho_lb = np.array(hard_constraints['coef_set'].lb)
        rho_ub = np.array(hard_constraints['coef_set'].ub)
        
        if '(Intercept)' in hard_constraints['coef_set'].variable_names:
            intercept_ind = hard_constraints['coef_set'].variable_names.index('(Intercept)')
            variable_ind = [i for i in range(0, P) if not i == intercept_ind]
            hard_constraints['max_offset'] = max(abs(rho_lb[intercept_ind]), abs(rho_lb[intercept_ind]))
            hard_constraints['max_coefficient'] = max(np.max(abs(rho_lb[variable_ind])), np.max(abs(rho_ub[variable_ind])))
        else:
            hard_constraints['max_offset'] = 0
            hard_constraints['max_coefficient'] = max(np.max(abs(rho_lb)), np.max(abs(rho_ub)))
        
        hard_constraints['max_L0_value'] = hard_constraints['L0_max']
    
    return hard_constraints
