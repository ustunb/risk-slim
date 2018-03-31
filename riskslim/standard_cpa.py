import time
import numpy as np
from cplex import Cplex, SparsePair, infinity as CPX_INFINITY
from .bound_tightening import chained_updates, chained_updates_for_lp
from .default_settings import DEFAULT_CPA_SETTINGS
from .helper_functions import get_or_set_default, get_rho_string, print_log
from .solution_classes import SolutionPool


def run_standard_cpa(cpx, cpx_indices, compute_loss, compute_loss_cut, settings = DEFAULT_CPA_SETTINGS, print_flag = False):

    assert isinstance(cpx, Cplex)
    assert isinstance(cpx_indices, dict)
    assert isinstance(settings, dict)
    assert callable(compute_loss)
    assert callable(compute_loss_cut)

    if print_flag:
        print_from_function = lambda msg: print_log(msg)
    else:
        print_from_function = lambda msg: None

    settings = get_or_set_default(settings, 'type', 'cvx')
    settings = get_or_set_default(settings, 'update_bounds', True)
    settings = get_or_set_default(settings, 'display_progress', True)
    settings = get_or_set_default(settings, 'save_progress', True)
    settings = get_or_set_default(settings, 'max_coefficient_gap', 0.5)
    settings = get_or_set_default(settings, 'max_tolerance', 0.00001)
    settings = get_or_set_default(settings, 'max_iterations', 10000)
    settings = get_or_set_default(settings, 'max_runtime', 100.0)
    settings = get_or_set_default(settings, 'max_runtime_per_iteration', 10000.0)
    settings = get_or_set_default(settings, 'max_cplex_time_per_iteration', 60.0)

    rho_idx = cpx_indices["rho"]
    loss_idx = cpx_indices["loss"]
    alpha_idx = cpx_indices["alpha"]
    cut_idx = loss_idx + rho_idx
    objval_idx = cpx_indices["objval"]
    L0_idx = cpx_indices["L0_norm"]

    if len(alpha_idx) == 0:
        get_alpha = lambda: np.array([])
    else:
        get_alpha = lambda: np.array(cpx.solution.get_values(alpha_idx))

    if type(loss_idx) is list and len(loss_idx) == 1:
        loss_idx = loss_idx[0]

    C_0_alpha = np.array(cpx_indices['C_0_alpha'])
    nnz_ind = np.flatnonzero(C_0_alpha)
    C_0_nnz = C_0_alpha[nnz_ind]

    if settings['update_bounds']:

        bounds = {
            'loss_min': cpx.variables.get_lower_bounds(loss_idx),
            'loss_max': cpx.variables.get_upper_bounds(loss_idx),
            'objval_min': cpx.variables.get_lower_bounds(objval_idx),
            'objval_max': cpx.variables.get_upper_bounds(objval_idx),
            'L0_min': cpx.variables.get_lower_bounds(L0_idx),
            'L0_max': cpx.variables.get_upper_bounds(L0_idx),
            }

        if settings['type'] == 'cvx':
            vtypes = 'C'
            update_problem_bounds = lambda bounds, lb, ub: chained_updates_for_lp(bounds, C_0_nnz, ub, lb)
        elif settings['type'] is 'ntree':
            vtypes = cpx.variables.get_types(rho_idx)
            update_problem_bounds = lambda bounds, lb, ub: chained_updates(bounds, C_0_nnz, ub, lb)

    else:
        update_problem_bounds = lambda bounds: None

    objval = 0.0
    upperbound = CPX_INFINITY
    lowerbound = 0.0

    n_iterations = 0
    simplex_iteration = 0
    stop_reason = 'aborted:reached_max_cuts'
    max_runtime = settings['max_runtime']
    remaining_total_time = max_runtime
    solutions = []
    objvals = []

    progress_stats = {
        'upperbounds': [],
        'lowerbounds': [],
        'simplex_iterations': [],
        'cut_times': [],
        'total_times': []
        }

    run_start_time = time.time()
    while n_iterations < settings['max_iterations']:

        iteration_start_time = time.time()
        current_timelimit = min(remaining_total_time, settings['max_cplex_time_per_iteration'])
        cpx.parameters.timelimit.set(float(current_timelimit))
        cpx.solve()
        solution_status = cpx.solution.status[cpx.solution.get_status()]

        # get solution
        if solution_status in ('optimal', 'optimal_tolerance', 'MIP_optimal'):
            rho = np.array(cpx.solution.get_values(rho_idx))
            alpha = get_alpha()
            simplex_iteration = int(cpx.solution.progress.get_num_iterations())
        else:
            stop_reason = solution_status
            print_from_function('stopping standard CPA (status: %s)' % solution_status)
            break

        # compute cut
        cut_start_time = time.time()
        loss_value, loss_slope = compute_loss_cut(rho)
        cut_lhs = [float(loss_value - loss_slope.dot(rho))]
        cut_constraint = [SparsePair(ind = cut_idx, val = [1.0] + (-loss_slope).tolist())]
        cut_name = ["ntree_cut_%d" % n_iterations]
        cut_time = time.time() - cut_start_time

        # compute objective bounds
        objval = float(loss_value + alpha.dot(C_0_alpha))
        upperbound = min(upperbound, objval)
        lowerbound = cpx.solution.get_objective_value()
        relative_gap = (upperbound - lowerbound)/(upperbound + np.finfo('float').eps)
        bounds = update_problem_bounds(bounds, lb = lowerbound, ub = upperbound)

        #store solutions
        solutions.append(rho)
        objvals.append(objval)

        # update run stats
        n_iterations += 1
        current_time = time.time()
        total_time = current_time - run_start_time
        iteration_time = current_time - iteration_start_time
        cplex_time = iteration_time - cut_time
        remaining_total_time = max(max_runtime - total_time, 0.0)

        # print information
        if settings['display_progress']:
            print_from_function("cuts = %d \t UB = %.5f \t LB = %.5f \t GAP = %.5f%%" % (n_iterations, upperbound, lowerbound, 100.0 * relative_gap))
            print_from_function('rho:%s\n' % get_rho_string(rho, vtypes))

        #save progress
        if settings['save_progress']:
            progress_stats['upperbounds'].append(upperbound)
            progress_stats['lowerbounds'].append(lowerbound)
            progress_stats['total_times'].append(total_time)
            progress_stats['cut_times'].append(cut_time)
            progress_stats['simplex_iterations'].append(simplex_iteration)

        # check termination conditions
        if len(solutions) >= 2:
            prior_rho = solutions[-2]
            coefficient_gap = np.abs(np.max(rho - prior_rho))
            if np.all(np.round(rho) == np.round(prior_rho)) and coefficient_gap < settings['max_coefficient_gap']:
                stop_reason = 'aborted:coefficient_gap_within_tolerance'
                print_from_function('stopping standard CPA')
                print_from_function('status: change in coefficient is within tolerance of %1.4f (%1.4f))' %
                                    (settings['max_coefficient_gap'], coefficient_gap))
                break

        if relative_gap < settings['max_tolerance']:
            stop_reason = 'converged:gap_within_tolerance'
            print_from_function('stopping standard CPA')
            print_from_function('status: optimality gap is within tolerance of %1.1f%% (%1.1f%%))' %
                                (100 * settings['max_tolerance'], 100 * relative_gap))
            break

        if cplex_time > settings['max_cplex_time_per_iteration']:
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('stopping standard CPA')
            print_from_function('status: reached max training time per iteration of %d secs (%d secs)' %
                                (settings['max_cplex_time_per_iteration'], cplex_time))

            break

        if iteration_time > settings['max_runtime_per_iteration']:
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('stopping standard CPA')
            print_from_function('status: reached max runtime time per iteration of %d secs (%d secs)' %
                                (settings['max_runtime_per_iteration'], iteration_time))
            break

        if (total_time > settings['max_runtime']) or (remaining_total_time == 0.0):
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('stopping standard CPA')
            print_from_function('status: reached max runtime time per iteration of %d secs (%d secs)' %
                                (settings['max_runtime'], total_time))
            break

        # switch bounds
        if settings['update_bounds']:
            cpx.variables.set_lower_bounds(L0_idx, bounds['L0_min'])
            cpx.variables.set_upper_bounds(L0_idx, bounds['L0_max'])
            cpx.variables.set_lower_bounds(loss_idx, bounds['loss_min'])
            cpx.variables.set_upper_bounds(loss_idx, bounds['loss_max'])
            cpx.variables.set_lower_bounds(objval_idx, bounds['objval_min'])
            cpx.variables.set_upper_bounds(objval_idx, bounds['objval_max'])

        # add loss cut
        cpx.linear_constraints.add(lin_expr = cut_constraint,
                                   senses = ["G"],
                                   rhs = cut_lhs,
                                   names = cut_name)

    #collect stats
    stats = {
        'solution': rho,
        'stop_reason': stop_reason,
        'n_iterations': n_iterations,
        'simplex_iteration': simplex_iteration,
        'objval': objval,
        'upperbound': upperbound,
        'lowerbound': lowerbound,
        'cut_time': cut_time,
        'total_time': total_time,
        'cplex_time': total_time - cut_time,
    }

    if settings['update_bounds']:
        stats.update(bounds)

    if settings['save_progress']:
        stats.update(progress_stats)
        stats['cplex_times'] = (np.array(stats['total_times']) - np.array(stats['cut_times'])).tolist()
        stats['objvals'] = objvals
        stats['solutions'] = solutions

    #collect cuts
    idx = range(cpx_indices['n_constraints'], cpx.linear_constraints.get_num(), 1)
    cuts = {
        'coefs': cpx.linear_constraints.get_rows(idx),
        'lhs': cpx.linear_constraints.get_rhs(idx)
    }

    #create solution pool
    if settings['type'] == 'ntree':
        C_0_rho = np.array(cpx_indices['C_0_rho'])
        for i in range(cpx.solution.pool.get_num()):
            rho = np.array(cpx.solution.pool.get_values(i, rho_idx))
            objval = compute_loss(rho) + np.sum(C_0_rho[np.flatnonzero(np.array(rho))])
            solutions.append(rho)
            objvals.append(objval)

    pool = SolutionPool(len(cpx_indices["rho"]))
    if len(objvals) > 0:
        pool.add(objvals, solutions)


    return stats, cuts, pool
