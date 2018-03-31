# THIS FILE CONTAINS CODE TO RUN ROUNDING METHODS FROM LARS_ELASTICNET MODELS
# to run: CALL script execute_script.py
# to debug: CALL script using test_onetree_script.py
import time
import numpy as np
import pandas as pd
from pprint import pprint
from rpy2.robjects import pandas2ri, r as rcmd
from risk_slim.SolutionPool import SolutionPool
from risk_slim.pipeline_helper_functions import print_log, load_or_set_default, load_hard_constraints, load_matlab_data, \
    load_loss_functions, get_accuracy_stats

global run_name, run_dir, data_dir

print_log('RUNNING FILE: rounding_methods_script.py')
print_log('run_name: %s' % run_name)
print_log('run_dir: %s' % run_dir)
print_log('data_dir: %s' % data_dir)
print_log('BLAS LINKAGE')
np.__config__.show()

#### LOAD HELPER FUNCTIONS
print_log('LOADING FUNCTIONS')


def load_from_disk(value_name, default_value = None, load_dir = run_dir, load_header = run_name):
    loaded_value = load_or_set_default(value_name, default_value, load_dir, load_header, print_flag = True)
    return loaded_value


def get_rounding_indices(solution,
                         feature_order,
                         L0_reg_ind = None,
                         L0_max = None):
    if L0_reg_ind is None:
        L0_reg_ind = np.ones_like(solution, dtype = 'bool')

    if L0_max is None:
        L0_max = np.sum(L0_reg_ind)

    ordered_indices = np.array(
            [j for j in np.argsort(feature_order).tolist() if j not in np.flatnonzero(~L0_reg_ind).tolist()])
    round_indices = np.concatenate((ordered_indices, np.flatnonzero(~L0_reg_ind)))
    drop_indices = ordered_indices[L0_max + 1:]
    round_indices = np.require(round_indices, dtype = 'int')
    drop_indices = np.require(drop_indices, dtype = 'int')
    return round_indices, drop_indices


def round_pool_solutions(pool,
                         hard_constraints,
                         use_capped_rounding = True,
                         use_scaled_rounding = True,
                         compute_objective_value = True):
    L0_max = hard_constraints['L0_max']
    L0_reg_ind = np.isnan(hard_constraints['coef_set'].C_0j)
    rho_ub = hard_constraints['coef_set'].ub
    rho_lb = hard_constraints['coef_set'].lb

    rounded_pool = SolutionPool(pool.P)
    for rho in pool.solutions:

        rho = np.copy(rho)
        feature_order = np.argsort(-abs(rho))
        round_indices, drop_indices = get_rounding_indices(rho, feature_order = feature_order, L0_reg_ind = L0_reg_ind,
                                                           L0_max = L0_max)

        if use_capped_rounding:
            rho_capped = np.round(rho)
            rho_capped = np.minimum(np.maximum(rho_capped, rho_lb), rho_ub)
            rho_capped[drop_indices] = 0.0
            capped_objval = get_objval(rho_capped) if compute_objective_value else np.nan
            rounded_pool = rounded_pool.add(objvals = capped_objval, solutions = rho_capped)

        elif use_scaled_rounding:
            rho_scaled = np.copy(rho)
            rho_scaled[drop_indices] = 0
            pos_coef_ind = rho_scaled > 0
            neg_coef_ind = rho_scaled < 0

            if any(pos_coef_ind) and any(neg_coef_ind):
                pos_scaling_factor = np.min(rho_ub[pos_coef_ind] / rho_scaled[pos_coef_ind])
                neg_scaling_factor = np.min(rho_lb[neg_coef_ind] / rho_scaled[neg_coef_ind])
                scaling_factor = min(pos_scaling_factor, neg_scaling_factor)
            elif any(pos_coef_ind):
                scaling_factor = np.min(rho_ub[pos_coef_ind] / rho_scaled[pos_coef_ind])
            elif any(neg_coef_ind):
                scaling_factor = np.min(rho_ub[neg_coef_ind] / rho_scaled[neg_coef_ind])
            else:
                scaling_factor = 1.0

            rho_scaled = np.round(rho_scaled * scaling_factor)
            rho_scaled = np.minimum(np.maximum(rho_scaled, rho_lb), rho_ub)
            scaled_objval = get_objval(rho_scaled) if compute_objective_value else np.nan
            rounded_pool = rounded_pool.add(objvals = scaled_objval, solutions = rho_scaled)

    return rounded_pool


def sequential_round_pool_solutions(pool):
    new_pool = SolutionPool(pool.P)
    total_runtime = 0.0
    for rho in pool.solutions:
        this_runtime = time.time()
        solution, objval, _ = sequential_rounding_greedy(np.copy(rho))
        total_runtime += time.time() - this_runtime
        new_pool = new_pool.add(objvals = objval, solutions = solution)

    return new_pool, total_runtime, len(new_pool)


def greedy_1_opt_on_pool_solutions(pool, active_set_flag = False):
    new_pool = SolutionPool(pool.P)
    total_runtime = 0.0

    for rho in pool.solutions:
        this_runtime = time.time()
        polished_solution, _, polished_objval = greedy_1_opt(rho = np.copy(rho), active_set_flag = active_set_flag)
        total_runtime += time.time() - this_runtime
        new_pool = new_pool.add(objvals = polished_objval, solutions = polished_solution)

    return new_pool, total_runtime, len(new_pool)


def sequential_rounding_greedy(rho):
    rho = np.copy(rho)
    early_stop_flag = False
    dimensions_to_round = np.flatnonzero(np.mod(rho, 1)).tolist()

    if len(dimensions_to_round) == 0:
        best_objval = compute_loss(rho)
        return rho, best_objval, early_stop_flag

    P = rho.shape[0]
    floor_is_zero = np.equal(np.floor(rho), 0)
    ceil_is_zero = np.equal(np.ceil(rho), 0)
    dist_from_start_to_ceil = np.ceil(rho) - rho
    dist_from_start_to_floor = np.floor(rho) - rho
    scores = Z.dot(rho)
    best_objval = float('inf')

    while len(dimensions_to_round) > 0:

        objvals_at_floor = np.full(P, np.nan)
        objvals_at_ceil = np.full(P, np.nan)
        current_penalty = get_L0_penalty(rho)

        for dim_idx in dimensions_to_round:

            # scores go from center to ceil -> center + dist_from_start_to_ceil
            scores += dist_from_start_to_ceil[dim_idx] * Z[:, dim_idx]
            objvals_at_ceil[dim_idx] = compute_loss_from_scores_real(scores)

            # move from ceil to floor => -1*Z_j
            scores -= Z[:, dim_idx]
            objvals_at_floor[dim_idx] = compute_loss_from_scores_real(scores)

            # scores go from floor to center -> floor - dist_from_start_to_floor
            scores -= dist_from_start_to_floor[dim_idx] * Z[:, dim_idx]
            # assert(np.all(np.isclose(scores, base_scores)))

            if ceil_is_zero[dim_idx]:
                objvals_at_ceil[dim_idx] -= C_0[dim_idx]
            elif floor_is_zero[dim_idx]:
                objvals_at_floor[dim_idx] -= C_0[dim_idx]

        # adjust for penalty value
        objvals_at_ceil += current_penalty
        objvals_at_floor += current_penalty
        best_objval_at_ceil = np.nanmin(objvals_at_ceil)
        best_objval_at_floor = np.nanmin(objvals_at_floor)

        if best_objval_at_ceil <= best_objval_at_floor:
            best_objval = best_objval_at_ceil
            best_dim = np.nanargmin(objvals_at_ceil)
            rho[best_dim] += dist_from_start_to_ceil[best_dim]
            scores += dist_from_start_to_ceil[best_dim] * Z[:, best_dim]
        else:
            best_objval = best_objval_at_floor
            best_dim = np.nanargmin(objvals_at_floor)
            rho[best_dim] += dist_from_start_to_floor[best_dim]
            scores += dist_from_start_to_floor[best_dim] * Z[:, best_dim]

        # assert(np.all(np.isclose(scores, Z.dot(rho))))
        dimensions_to_round.remove(best_dim)

    return rho, best_objval, early_stop_flag


def greedy_1_opt(rho, active_set_flag = False, print_flag = False):
    # initialize key variables
    n_iterations = 0
    P = rho.shape[0]
    MAX_ITERATIONS = 500
    MIN_IMPROVEMENT_PER_STEP = float(1e-10)

    if active_set_flag:
        active_set = np.flatnonzero(rho).tolist() + np.flatnonzero(~L0_reg_ind).tolist()
        full_search_dimensions = np.unique(np.sort(active_set)).tolist()
    else:
        full_search_dimensions = range(0, P)

    rho = np.require(np.require(rho, dtype = np.int_), dtype = np.float_)

    base_scores = Z.dot(rho)
    base_loss = compute_loss_from_scores(base_scores)
    base_objval = base_loss + get_L0_penalty(rho)
    search_dimensions = full_search_dimensions
    keep_searching = True

    while keep_searching and n_iterations < MAX_ITERATIONS:

        # compute the best objective value / step size in each dimension
        best_objval_by_dim = np.array([np.nan] * P)
        best_coef_by_dim = np.array([np.nan] * P)

        for k in search_dimensions:
            feasible_coefs_for_dim = np.arange(int(rho_lb[k]), int(rho_ub[k]) + 1)  # TODO CHANGE THIS
            objvals = compute_objvals_at_dim(k, feasible_coefs_for_dim, rho, base_scores, base_loss, C_0)
            objvals[np.where(objvals == base_objval)] = np.inf
            best_objval_by_dim[k] = np.nanmin(objvals)
            best_coef_by_dim[k] = feasible_coefs_for_dim[np.nanargmin(objvals)]

        # check if there exists a move that yields an improvement
        # print_log('')
        # print_log('ITERATION %d' % n_iterations)
        # print_log('search dimensions has %d/%d dimensions' % (len(search_dimensions), P))
        # print_log(search_dimensions)
        # print_log('best_objval_by_dim')
        # print_log(["{0:0.20f}".format(i) for i in best_objval_by_dim])

        next_objval = np.nanmin(best_objval_by_dim)
        # print_log('IMPROVEMENT: %1.20f' % (base_objval - next_objval))

        if next_objval < (base_objval - MIN_IMPROVEMENT_PER_STEP):
            # take the best step in the best direction
            step_dim = int(np.nanargmin(best_objval_by_dim))

            if print_flag:
                print_log("improving objective value from %1.16f to %1.16f" % (base_objval, next_objval))
                print_log(
                    "changing rho[%d] from %1.0f to %1.0f" % (step_dim, rho[step_dim], best_coef_by_dim[step_dim]))

            rho[step_dim] = best_coef_by_dim[step_dim]

            # recompute base objective value/loss/scores
            base_objval = next_objval
            base_loss = base_objval - get_L0_penalty(rho)
            base_scores = Z.dot(rho)

            # remove the current best direction from the set of directions to explore
            search_dimensions = full_search_dimensions
            search_dimensions.remove(step_dim)
            n_iterations += 1
        else:
            keep_searching = False

    if print_flag:
        print_log("completed %d iterations" % n_iterations)
        print_log("current: %1.10f < best possible: %1.10f" % (base_objval, next_objval))

    return rho, base_loss, base_objval


def compute_objvals_at_dim(dim_index,
                           feasible_coef_values,
                           base_rho,
                           base_scores,
                           base_loss,
                           C_0):
    """
    finds the value of rho[j] in feasible_coef_values that minimizes log_loss(rho) + C_0j
    :param dim_index:
    :param feasible_coef_values:
    :param base_rho:
    :param base_scores:
    :param base_loss:
    :param C_0:
    :return:
    """

    # copy stuff because ctypes
    scores = np.copy(base_scores)

    # initialize parameters
    base_coef_value = base_rho[dim_index]
    base_index = np.where(feasible_coef_values == base_coef_value)[0]
    loss_at_coef_value = np.array([np.nan] * len(feasible_coef_values))
    loss_at_coef_value[base_index] = np.float(base_loss)
    Z_dim = Z[:, dim_index]

    # start by moving forward
    forward_indices = np.where(base_coef_value <= feasible_coef_values)[0]
    forward_step_sizes = np.diff(feasible_coef_values[forward_indices] - base_coef_value)
    n_forward_steps = len(forward_step_sizes)
    stop_after_first_forward_step = False

    best_loss = base_loss
    total_distance_from_base = 0

    for i in range(0, n_forward_steps):
        scores += forward_step_sizes[i] * Z_dim
        total_distance_from_base += forward_step_sizes[i]
        current_loss = compute_loss_from_scores(scores)
        if current_loss >= best_loss:
            stop_after_first_forward_step = (i == 0)
            break
        loss_at_coef_value[forward_indices[i + 1]] = current_loss
        best_loss = current_loss

    # if the first step forward didn't lead to a decrease in loss, then move backwards
    move_backward = stop_after_first_forward_step or n_forward_steps == 0

    if move_backward:

        # compute backward steps
        backward_indices = np.flipud(np.where(feasible_coef_values <= base_coef_value)[0])
        backward_step_sizes = np.diff(feasible_coef_values[backward_indices] - base_coef_value)
        n_backward_steps = len(backward_step_sizes)

        # correct size of first backward step if you took 1 step forward
        if n_backward_steps > 0 and n_forward_steps > 0:
            backward_step_sizes[0] = backward_step_sizes[0] - forward_step_sizes[0]

        best_loss = base_loss

        for i in range(0, n_backward_steps):
            scores += backward_step_sizes[i] * Z_dim
            total_distance_from_base += backward_step_sizes[i]
            current_loss = compute_loss_from_scores(scores)
            if current_loss >= best_loss:
                break
            loss_at_coef_value[backward_indices[i + 1]] = current_loss
            best_loss = current_loss

    # at this point scores == base_scores + step_distance*Z_dim
    # assert(all(np.isclose(scores, base_scores + total_distance_from_base * Z_dim)))

    # compute objective values by adding penalty values to all other indices
    other_dim_idx = np.where(dim_index != np.arange(0, P))[0]
    other_dim_penalty = np.sum(C_0[other_dim_idx] * (base_rho[other_dim_idx] != 0))
    objval_at_coef_values = loss_at_coef_value + other_dim_penalty

    if C_0[dim_index] > 0.0:

        # increase objective value at every non-zero coefficient value by C_0j
        nonzero_coef_idx = np.flatnonzero(feasible_coef_values)
        objval_at_coef_values[nonzero_coef_idx] = objval_at_coef_values[nonzero_coef_idx] + C_0[dim_index]

        # compute value at coef[j] == 0 if needed
        zero_coef_idx = np.where(feasible_coef_values == 0)[0]
        if np.isnan(objval_at_coef_values[zero_coef_idx]):
            # steps_from_here_to_zero: step_from_here_to_base + step_from_base_to_zero
            # steps_from_here_to_zero: -step_from_base_to_here + -step_from_zero_to_base
            steps_to_zero = -(base_coef_value + total_distance_from_base)
            scores += steps_to_zero * Z_dim
            objval_at_coef_values[zero_coef_idx] = compute_loss_from_scores(scores) + other_dim_penalty
            # assert(all(np.isclose(scores, base_scores - base_coef_value * Z_dim)))

    # return objective value at feasible coefficients
    return objval_at_coef_values


SMALLEST_C0_VALUE = 1e-6

#### READ SETTINGS FROM DISK
#### READ SETTINGS FROM DISK
print_log('READING SETTINGS FROM DISK')

comp_name = load_from_disk("comp_name", "berkmac")
lars_elasticnet_file = load_from_disk("lars_elasticnet_file", 'mammo_F_K05N01_RSPP_processed.RData')
results_file_suffix = load_from_disk("results_file_suffix", "rounded_RSPP_results")

settings = {
    'save_LR': True,
    'save_CR': True,
    'save_SR': True,
    'save_SRDCD': True,
    'save_LRDCD': True,
    'save_CRDCD': True,
    'filter_hcon_id': "NONE",
    'filter_xtra_id': "O00C5T00",
    }
settings = {key: load_from_disk(key, settings[key]) for key in settings}

##### LOAD CTS SOLUTIONS
print_log('LOADING FROM DISK %s' % (run_dir + lars_elasticnet_file))

# load results and models from RData file
pandas2ri.activate()
rcmd['load'](run_dir + lars_elasticnet_file)
results_df = pandas2ri.PandasDataFrame(rcmd('results_df'))
results_df = results_df[results_df.method_name.isin(['lars_elasticnet'])]

if settings['filter_xtra_id'] != 'NONE':
    results_df = results_df[results_df.xtra_id.isin([settings['filter_xtra_id']])]

if settings['filter_hcon_id'] != 'NONE':
    results_df = results_df[results_df.hcon_id.isin([settings['filter_hcon_id']])]

print_models = rcmd('data.frame(do.call(rbind, print_models[results_df$print_model_id]))')
print_log('LOADED ELASTICNET MODELS FROM FILE: %s' % lars_elasticnet_file)
pandas2ri.deactivate()

# setup loops
loop_columns = [
    'data_name',
    'xtra_id',
    'hcon_id',
    'fold_id',
    'inner_fold_id',
    'fold',
    'sample_weight_id',
    'w_pos',
    'w_neg'
    ]

base_accuracy_columns = [
    'false_negatives',
    'false_positives',
    'true_negatives',
    'true_positives'
    ]

accuracy_columns = ['train_' + c for c in base_accuracy_columns] + \
                   ['valid_' + c for c in base_accuracy_columns] + \
                   ['test_' + c for c in base_accuracy_columns]

id_columns = [col_name for col_name in results_df.columns.values if 'parameter' in col_name]
filter_columns = loop_columns + id_columns + ['print_model_id', 'runtime']

# create columns
base_results_df = results_df[filter_columns].sort_values(by = loop_columns, axis = 0, inplace = False,
                                                         ascending = False)
for cname in accuracy_columns:
    base_results_df[cname] = float('nan')

# setup rounding methods
all_rounding_methods = []
all_method_suffixes = ['SR', 'CR', 'LR', 'SRDCD', 'LRDCD', 'CRDCD']
for method_suffix in all_method_suffixes:
    method_name = 'RSLP_%s' % method_suffix
    setting_name = 'save_%s' % method_suffix
    if setting_name in settings and settings[setting_name]:
        all_rounding_methods += [method_name]

initial_rounding_methods = []
if 'RSLP_SR' in all_rounding_methods or 'RSLP_SRDCD' in all_rounding_methods:
    initial_rounding_methods += ['sequential']

if 'RSLP_CR' in all_rounding_methods or 'RSLP_CRDCD' in all_rounding_methods:
    initial_rounding_methods += ['capped']

if 'RSLP_LR' in all_rounding_methods or 'RSLP_LRDCD' in all_rounding_methods:
    initial_rounding_methods += ['scaled']

#### ROUND MODELS USING ROUNDING METHODS
all_results = []
all_models = []
loop_info_df = base_results_df[loop_columns].drop_duplicates(loop_columns)
n_loops = loop_info_df.shape[0]

for ind in range(n_loops):

    loop_info = loop_info_df.iloc[ind].to_dict()
    loop_info['max_coefficient'] = 5
    loop_info['max_offset'] = -1
    loop_info['max_L0_value'] = -1

    # hard constraints
    loop_info['has_custom_coefficient_set'] = (loop_info['xtra_id'].isalnum() and
                                               'O' in loop_info['xtra_id'] and
                                               'C' in loop_info['xtra_id'] and
                                               'T' in loop_info['xtra_id'])

    if loop_info['has_custom_coefficient_set']:
        tmp = loop_info['xtra_id'].split('T')
        max_L0_value = int(tmp[1])
        tmp = tmp[0]
        if max_L0_value == 0:
            max_L0_value = -1

        tmp = tmp.split('C')
        max_coefficient = int(tmp[1])
        tmp = tmp[0]

        tmp = tmp.split('O')
        max_offset = int(tmp[1])
        if max_offset == 0:
            max_offset = -1

        loop_info['max_L0_value'] = max_L0_value
        loop_info['max_coefficient'] = max_coefficient
        loop_info['max_offset'] = max_offset

    # print loop information
    print_log('STARTING ROUNDING LOOP #%d/%d' % (ind + 1, n_loops))
    pprint(loop_info)

    # data
    data = load_matlab_data(data_file_name = data_dir + loop_info['data_name'] + "_processed.mat",
                            fold_id = loop_info['fold_id'],
                            fold_num = loop_info['fold'],
                            inner_fold_id = loop_info['inner_fold_id'],
                            sample_weight_id = loop_info['sample_weight_id'])

    Z = data['X'] * data['Y']
    N, P = Z.shape
    pos_ind = data['Y'].flatten() == 1
    variable_names = list(data['X_headers'])

    # weights
    training_weights = data['sample_weights'].flatten()
    training_weights = len(training_weights) * (training_weights / sum(training_weights))
    w_total = loop_info['w_pos'] + loop_info['w_neg']
    w_pos = 2.00 * (loop_info['w_pos'] / w_total)
    w_neg = 2.00 * (loop_info['w_neg'] / w_total)
    training_weights[pos_ind] *= w_pos
    training_weights[~pos_ind] *= w_neg

    # hard constraints
    hard_constraints = load_hard_constraints(data,
                                             data_file_name = data_dir + loop_info['data_name'] + "_processed.mat",
                                             hcon_id = loop_info['hcon_id'],
                                             use_custom_coefficient_set = loop_info['has_custom_coefficient_set'],
                                             max_coefficient = loop_info['max_coefficient'],
                                             max_offset = loop_info['max_offset'],
                                             max_L0_value = loop_info['max_L0_value'])

    rho_lb = np.array(hard_constraints['coef_set'].lb)
    rho_ub = np.array(hard_constraints['coef_set'].ub)
    C_0 = np.array(hard_constraints['coef_set'].C_0j)
    L0_reg_ind = np.isnan(C_0)
    c0_value = SMALLEST_C0_VALUE
    C_0[L0_reg_ind] = c0_value

    # function handles
    (compute_loss,
     _,
     compute_loss_from_scores,
     compute_loss_real,
     _,
     compute_loss_from_scores_real) = load_loss_functions(Z,
                                                          loss_computation = 'lookup',
                                                          rho_ub = hard_constraints['coef_set'].ub,
                                                          rho_lb = hard_constraints['coef_set'].lb,
                                                          L0_reg_ind = np.isnan(hard_constraints['coef_set'].C_0j),
                                                          L0_max = hard_constraints['L0_max'],
                                                          weights = training_weights)

    is_integer = lambda rho: np.array_equal(rho, np.require(rho, dtype = np.int_))
    cast_to_integer = lambda rho: np.require(np.require(rho, dtype = np.int_), dtype = rho.dtype)
    get_L0_norm = lambda rho: np.count_nonzero(rho[L0_reg_ind])
    get_L0_penalty = lambda rho: np.sum(C_0[L0_reg_ind] * (rho[L0_reg_ind] != 0.0))
    get_objval = lambda rho: compute_loss(rho) + get_L0_penalty(rho)
    get_objval_real = lambda rho: compute_loss_real(rho) + get_L0_penalty(rho)

    ####ROUNDING
    loop_results = base_results_df.copy()
    filter_ind = (
            (loop_results.data_name == loop_info['data_name']) &
            (loop_results.fold_id == loop_info['fold_id']) &
            (loop_results.inner_fold_id == loop_info['inner_fold_id']) &
            (loop_results.xtra_id == loop_info['xtra_id']) &
            (loop_results.hcon_id == loop_info['hcon_id']) &
            (loop_results.sample_weight_id == loop_info['sample_weight_id']) &
            (loop_results.w_pos == loop_info['w_pos']) &
            (loop_results.w_neg == loop_info['w_neg']) &
            (loop_results.fold == loop_info['fold'])
    )
    loop_results = loop_results[filter_ind].reset_index(drop = True)

    # setup continuous solution pool
    cts_models = print_models.loc[loop_results['print_model_id']].as_matrix()
    cts_pool = SolutionPool(P)
    cts_pool = cts_pool.add(solutions = cts_models, objvals = [np.nan] * cts_models.shape[0])
    print_log('%d SOLUTIONS IN CTS POOL' % len(cts_pool))

    for rounding_method in initial_rounding_methods:

        if rounding_method == 'capped':
            print_log('METHOD: CAPPED ROUNDING')
            initial_name = 'RSLP_CR'
        elif rounding_method == 'scaled':
            print_log('METHOD: SCALED ROUNDING')
            initial_name = 'RSLP_LR'
        elif rounding_method == 'sequential':
            print_log('METHOD: SEQUENTIAL ROUNDING')
            initial_name = 'RSLP_SR'
        else:
            raise ValueError('invalid rounding method')

        polished_name = initial_name + 'DCD'
        use_capped_rounding = rounding_method == 'capped'
        use_scaled_rounding = rounding_method == 'scaled'
        use_sequential_rounding = rounding_method == 'sequential'
        save_initial = initial_name in all_rounding_methods
        save_polished = save_initial and (polished_name in all_rounding_methods)

        # ROUND CONTINUOUS SOLUTIONS
        start_time = time.time()
        if use_sequential_rounding:
            rounded_pool, _, _ = sequential_round_pool_solutions(pool = cts_pool)
        else:
            rounded_pool = round_pool_solutions(pool = cts_pool,
                                                hard_constraints = hard_constraints,
                                                use_capped_rounding = use_capped_rounding,
                                                use_scaled_rounding = use_scaled_rounding,
                                                compute_objective_value = True)

        round_time = time.time() - start_time
        print_log('ROUNDED %d SOLUTIONS IN %1.2f SECONDS' % (len(rounded_pool), round_time))
        assert (len(rounded_pool) == len(cts_pool))
        assert (len(rounded_pool) == loop_results.shape[0])

        if save_initial:
            round_results = loop_results.copy()
            round_results['method_name'] = initial_name
            round_results['runtime'] += round_time / len(rounded_pool)
            round_results['model_size'] = [get_L0_norm(sol) for sol in rounded_pool.solutions]

            for n in range(len(rounded_pool)):
                accuracy_stats = get_accuracy_stats(rounded_pool.solutions[n], data)
                for stat_name in accuracy_stats:
                    round_results.loc[n, stat_name] = accuracy_stats[stat_name]

            assert (len(rounded_pool) == round_results.shape[0])
            all_results.append(round_results)
            all_models.append(np.copy(rounded_pool.solutions))
            print_log('APPENDED ROUNDED SOLUTIONS | method_name: %s' % round_results['method_name'][0])

        if save_polished:
            polished_pool, polish_time, _ = greedy_1_opt_on_pool_solutions(pool = rounded_pool,
                                                                           active_set_flag = True)

            round_time = time.time() - start_time
            print_log('POLISHED %d SOLUTIONS IN %1.2f SECONDS' % (len(polished_pool), polish_time))
            assert (len(polished_pool) == len(rounded_pool))
            assert (len(polished_pool) == loop_results.shape[0])
            assert (np.all(polished_pool.objvals <= rounded_pool))

            # COMPLETE LOOP RESULTS
            polish_results = loop_results.copy()
            polish_results['method_name'] = polished_name
            polish_results['runtime'] += round_time / len(polished_pool)
            polish_results['model_size'] = [get_L0_norm(sol) for sol in polished_pool.solutions]

            for n in range(len(polished_pool)):
                accuracy_stats = get_accuracy_stats(polished_pool.solutions[n], data, error_checking = False)
                for stat_name in accuracy_stats:
                    polish_results.loc[n, stat_name] = accuracy_stats[stat_name]

            assert (len(polished_pool) == polish_results.shape[0])
            all_results.append(polish_results)
            all_models.append(np.copy(polished_pool.solutions))
            print_log('APPENDED POLISHED SOLUTIONS | method_name: %s' % polish_results['method_name'][0])

#### PROCESS AND SAVE RESULTS FOR ROUNDING METHODS

all_results_df = pd.concat(objs = all_results,
                           axis = 0,
                           join = 'outer',
                           ignore_index = True,
                           copy = True)

save_loop_columns = ['data_name',
                     'fold_id',
                     'inner_fold_id',
                     'xtra_id',
                     'hcon_id',
                     'sample_weight_id',
                     'w_pos',
                     'w_neg']

parameter_columns = [col_name for col_name in all_results_df.columns.values if 'parameter' in col_name]

selected_columns = (save_loop_columns +
                    ['method_name'] +
                    parameter_columns +
                    ['fold', 'runtime', 'model_size'] +
                    accuracy_columns)

all_results_df = all_results_df.reindex(columns = selected_columns)

# models
all_models_df = np.vstack(all_models)
assert (all_models_df.shape[0] == all_results_df.shape[0])
save_loop_info_df = all_results_df[save_loop_columns].drop_duplicates(save_loop_columns)
n_save_files = save_loop_info_df.shape[0]

pandas2ri.activate()
for ind in range(n_save_files):

    loop_info = save_loop_info_df.iloc[ind].copy()

    # setup run
    run_name = '%s_F_%s_I_%s_W_%s_L_%s_X_%s_pos_%1.9f' % (loop_info['data_name'],
                                                          loop_info['fold_id'],
                                                          loop_info['inner_fold_id'],
                                                          loop_info['sample_weight_id'],
                                                          loop_info['hcon_id'],
                                                          loop_info['xtra_id'],
                                                          loop_info['w_pos'])

    loop_info['has_sample_weights'] = bool(loop_info['sample_weight_id'] != "NONE")
    loop_info['has_class_weights'] = bool(loop_info['w_pos'] != 1.00)
    loop_info['comp_name'] = comp_name
    loop_info['run_name'] = run_name
    loop_info['run_dir'] = run_dir
    loop_info['date'] = time.strftime('%m/%d/%Y')
    loop_info['data_file_name'] = data_dir + loop_info['data_name'] + '_processed.mat'
    loop_info['results_file_name'] = run_dir + run_name + '_' + results_file_suffix + '.RData'

    # variable_names
    data = load_matlab_data(data_file_name = loop_info['data_file_name'],
                            fold_id = loop_info['fold_id'],
                            fold_num = 0,
                            inner_fold_id = loop_info['inner_fold_id'],
                            sample_weight_id = loop_info['sample_weight_id'])

    # setup results and models
    save_row_ind = (
            (all_results_df.data_name == loop_info['data_name']) &
            (all_results_df.fold_id == loop_info['fold_id']) &
            (all_results_df.inner_fold_id == loop_info['inner_fold_id']) &
            (all_results_df.xtra_id == loop_info['xtra_id']) &
            (all_results_df.hcon_id == loop_info['hcon_id']) &
            (all_results_df.sample_weight_id == loop_info['sample_weight_id']) &
            (all_results_df.w_pos == loop_info['w_pos']) &
            (all_results_df.w_neg == loop_info['w_neg'])
    )

    results_R = all_results_df[save_row_ind]
    models_R = all_models_df[np.array(save_row_ind), :]

    # create R file
    rcmd.assign("info", loop_info)
    rcmd.assign("variable_names", list(data['X_headers']))
    rcmd.assign("results", results_R)
    rcmd.assign("models", models_R)

    rcmd("""
    models = split(models, row(models));
    models = lapply(models, function(x){setNames(x, variable_names)})

    output = list();
    all_methods = unique(results$method_name);
    for (method_name in all_methods){

        method_output = list();
        method_output$method_name = method_name;
        method_output$method_settings = list();

        ind = results$method_name == method_name;
        n_models = sum(ind);

        model_ids = sprintf('M%08d', seq(1, n_models));
        print_models = models[ind];
        debug_models = vector('list', n_models);
        names(print_models) = model_ids;
        names(debug_models) = model_ids;
        method_output$print_models = print_models;
        method_output$debug_models = debug_models;

        method_output$results_df = results[ind,];
        method_output$results_df$print_model_id = model_ids;
        method_output$results_df$debug_model_id = model_ids;
        method_output$total_runtime = sum(method_output$results_df$runtime);

        output[[method_name]] = method_output;
    }
    """)

    try:
        rcmd.save("output", "info", file = loop_info['results_file_name'])
        print_log("SAVED RESULTS TO FILE %s" % loop_info['results_file_name'])
    except:
        print_log("FAILED TO SAVE RESULTS")
pandas2ri.deactivate()

print_log("REACHED END OF rounding_methods_script.py")