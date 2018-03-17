import numpy as np
import time
from solution_classes import SolutionPool


def round_solution_pool(pool, constraints):
    """

    Parameters
    ----------
    pool
    constraints

    Returns
    -------

    """

    pool.distinct().sort()
    P = pool.P
    L0_reg_ind = np.isnan(constraints['coef_set'].C_0j)
    L0_max = constraints['L0_max']
    rounded_pool = SolutionPool(P)

    for solution in pool.solutions:
        # sort from largest to smallest coefficients
        feature_order = np.argsort([-abs(x) for x in solution])
        rounded_solution = np.zeros(shape=(1, P))
        l0_norm_count = 0
        for k in range(P):
            j = feature_order[k]
            if not L0_reg_ind[j]:
                rounded_solution[0, j] = np.round(solution[j], 0)
            elif l0_norm_count < L0_max:
                rounded_solution[0, j] = np.round(solution[j], 0)
                l0_norm_count += L0_reg_ind[j]

        rounded_pool.add(objvals=np.nan, solutions=rounded_solution)

    rounded_pool.distinct().sort()
    return rounded_pool


def sequential_round_solution_pool(pool,
                                   Z,
                                   C_0,
                                   compute_loss_from_scores_real,
                                   get_L0_penalty,
                                   max_runtime = float('inf'),
                                   max_solutions = float('inf'),
                                   objval_cutoff = float('inf'),
                                   L0_min = 0,
                                   L0_max = float('inf')):
    """

    Parameters
    ----------
    pool
    Z
    C_0
    max_runtime
    max_solutions
    objval_cutoff
    L0_min
    L0_max

    Returns
    -------

    """
    # quick return
    if len(pool) == 0:
        return pool, 0.0, 0

    P = pool.P
    total_runtime = 0.0

    #todo: filter out solutions that can only be rounded one way
    #rho is integer

    #if model size constraint is non-trivial, remove solutions that will violate the model size constraint
    if L0_min > 0 and L0_max < P:
        L0_reg_ind = np.isnan(C_0)
        def rounded_model_size_is_ok(rho):
            abs_rho = abs(rho)
            rounded_rho_l0_min = np.count_nonzero(np.floor(abs_rho[L0_reg_ind]))
            rounded_rho_l0_max = np.count_nonzero(np.ceil(abs_rho[L0_reg_ind]))
            return rounded_rho_l0_max >= L0_min and rounded_rho_l0_min <= L0_max

        pool = pool.removeInfeasible(rounded_model_size_is_ok)

    pool = pool.sort()
    rounded_pool = SolutionPool(P)
    n_to_round = int(min(max_solutions, len(pool)))

    #round solutions using sequential rounding
    for n in range(n_to_round):

        start_time = time.time()
        solution, objval, early_stop = sequential_rounding(rho = pool.solutions[n, ],
                                                           Z = Z,
                                                           C_0 = C_0,
                                                           compute_loss_from_scores_real = compute_loss_from_scores_real,
                                                           get_L0_penalty = get_L0_penalty,
                                                           objval_cutoff = objval_cutoff)
        total_runtime += time.time() - start_time

        if not early_stop:
            rounded_pool = rounded_pool.add(objvals = objval, solutions = solution)

        if total_runtime > max_runtime:
            break

    n_rounded = len(rounded_pool)
    rounded_pool = rounded_pool.sort()
    return rounded_pool, total_runtime, n_rounded


def discrete_descent_solution_pool(pool,
                                   Z,
                                   C_0,
                                   constraints,
                                   get_L0_penalty,
                                   compute_loss_from_scores,
                                   max_runtime = float('inf'),
                                   max_solutions = float('inf')):
    """
    runs DCD polishing for all solutions in the a solution pool
    can be stopped early using max_runtime or max_solutions

    Parameters
    ----------
    pool
    Z
    C_0
    constraints
    max_runtime
    max_solutions

    Returns
    -------
    new solution pool, total polishing time, and # of solutions polished
    """

    if len(pool) == 0:
        return pool, 0.0, 0

    rho_ub = constraints['coef_set'].ub
    rho_lb = constraints['coef_set'].lb
    total_runtime = 0.0
    pool = pool.sort()
    polished_pool = SolutionPool(pool.P)
    n_to_polish = min(max_solutions, len(pool))

    for n in range(n_to_polish):
        start_time = time.time()
        polished_solution, _, polished_objval = discrete_descent(rho = pool.solutions[n],
                                                                 Z = Z,
                                                                 C_0 = C_0,
                                                                 rho_ub = rho_ub,
                                                                 rho_lb = rho_lb,
                                                                 get_L0_penalty = get_L0_penalty,
                                                                 compute_loss_from_scores = compute_loss_from_scores)

        total_runtime += time.time() - start_time
        polished_pool = polished_pool.add(objvals = polished_objval, solutions = polished_solution)
        if total_runtime > max_runtime:
            break

    n_polished = len(polished_pool)
    polished_pool = polished_pool.append(pool).sort()
    return polished_pool, total_runtime, n_polished


def sequential_rounding(rho,
                        Z,
                        C_0,
                        compute_loss_from_scores_real,
                        get_L0_penalty,
                        objval_cutoff = float('Inf')):
    """
    :param rho: continuous solution st. rho_lb[i] <= rho[i] <= rho_ub[i]
    :param objval_cutoff: cutoff value at which we return null
    :return: rho: integer solution st. rho_lb[i] <= rho[i] <= rho_ub[i]
    """

    P = rho.shape[0]
    dimensions_to_round = np.flatnonzero(np.mod(rho, 1)).tolist()
    floor_is_zero = np.equal(np.floor(rho), 0)
    ceil_is_zero = np.equal(np.ceil(rho), 0)

    dist_from_start_to_ceil = np.ceil(rho) - rho
    dist_from_start_to_floor = np.floor(rho) - rho

    scores = Z.dot(rho)
    best_objval = float('inf')
    early_stop_flag = False

    while len(dimensions_to_round) > 0:

        objvals_at_floor = np.array([np.nan] * P)
        objvals_at_ceil = np.array([np.nan] * P)
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
        best_objval = min(best_objval_at_ceil, best_objval_at_floor)

        if best_objval > objval_cutoff:
            best_objval = float('nan')
            early_stop_flag = True
            break
        else:
            if best_objval_at_ceil <= best_objval_at_floor:
                best_dim = np.nanargmin(objvals_at_ceil)
                rho[best_dim] += dist_from_start_to_ceil[best_dim]
                scores += dist_from_start_to_ceil[best_dim] * Z[:, best_dim]
            else:
                best_dim = np.nanargmin(objvals_at_floor)
                rho[best_dim] += dist_from_start_to_floor[best_dim]
                scores += dist_from_start_to_floor[best_dim] * Z[:, best_dim]

        # assert(np.all(np.isclose(scores, Z.dot(rho))))
        dimensions_to_round.remove(best_dim)

    return rho, best_objval, early_stop_flag


def discrete_descent(rho,
                     Z,
                     C_0,
                     rho_ub,
                     rho_lb,
                     get_L0_penalty,
                     compute_loss_from_scores,
                     descent_dimensions = None,
                     print_flag = False):
    """
    given a initial feasible solution, rho, produces an improved solution that is 1-OPT
    (i.e. the objective value does not decrease by moving in any single dimension)
    at each iteration, the algorithm moves in the dimension that yields the greatest decrease in objective value
    the best step size is each dimension is computed using a directional search strategy that saves computation
    """
    #print_flag = False

    # initialize key variables
    MAX_ITERATIONS = 500
    MIN_IMPROVEMENT_PER_STEP = float(1e-10)
    n_iterations = 0
    P = rho.shape[0]
    rho = np.require(np.require(rho, dtype=np.int_), dtype=np.float_)
    if descent_dimensions is None:
        descent_dimensions = range(0, P)

    search_dimensions = descent_dimensions
    base_scores = Z.dot(rho)
    base_loss = compute_loss_from_scores(base_scores)
    base_objval = base_loss + get_L0_penalty(rho)
    keep_searching = True

    while keep_searching and n_iterations < MAX_ITERATIONS:

        # compute the best objective value / step size in each dimension
        best_objval_by_dim = np.array([np.nan] * P)
        best_coef_by_dim = np.array([np.nan] * P)

        for k in search_dimensions:
            feasible_coefs_for_dim = np.arange(int(rho_lb[k]), int(rho_ub[k]) + 1)  # TODO CHANGE THIS
            objvals = _compute_objvals_at_dim(Z = Z,
                                              C_0 = C_0,
                                              compute_loss_from_scores = compute_loss_from_scores,
                                              dim_index = k,
                                              feasible_coef_values = feasible_coefs_for_dim,
                                              base_rho = rho,
                                              base_scores = base_scores,
                                              base_loss = base_loss)

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

            # if print_flag:
            #     print_log("improving objective value from %1.16f to %1.16f" % (base_objval, next_objval))
            #     print_log(
            #         "changing rho[%d] from %1.0f to %1.0f" % (step_dim, rho[step_dim], best_coef_by_dim[step_dim]))

            rho[step_dim] = best_coef_by_dim[step_dim]

            # recompute base objective value/loss/scores
            base_objval = next_objval
            base_loss = base_objval - get_L0_penalty(rho)
            base_scores = Z.dot(rho)

            # remove the current best direction from the set of directions to explore
            search_dimensions = descent_dimensions
            search_dimensions.remove(step_dim)
            n_iterations += 1
        else:
            keep_searching = False

    # if print_flag:
    #     print_log("completed %d iterations" % n_iterations)
    #     print_log("current: %1.10f < best possible: %1.10f" % (base_objval, next_objval))

    return rho, base_loss, base_objval


def _compute_objvals_at_dim(Z,
                            C_0,
                            compute_loss_from_scores,
                            dim_index,
                            feasible_coef_values,
                            base_rho,
                            base_scores,
                            base_loss):
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
    P = base_rho.shape[0]
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

    for i in range(n_forward_steps):
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

        for i in range(n_backward_steps):
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


