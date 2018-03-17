import numpy as np
import time
from cplex import infinity as CPX_INFINITY, SparsePair
from .helper_functions import print_log, print_update, get_or_set_default
from .bound_tightening import chained_updates, chained_updates_for_lp
from .solution_classes import SolutionPool

# Standard CPA
def run_standard_cpa(MIP, indices, settings, compute_loss, compute_loss_cut, print_flag = True):

    if print_flag:
        print_from_function = lambda msg: print_log(msg)
    else:
        print_from_function = lambda: None

    settings = get_or_set_default(settings, 'type', 'cvx')
    settings = get_or_set_default(settings, 'update_bounds', True)
    settings = get_or_set_default(settings, 'display_progress', True)
    settings = get_or_set_default(settings, 'save_progress', True)
    settings = get_or_set_default(settings, 'max_tolerance', 0.00001)
    settings = get_or_set_default(settings, 'max_iterations', 10000)
    settings = get_or_set_default(settings, 'max_runtime', 100.0)
    settings = get_or_set_default(settings, 'max_runtime_per_iteration', 10000.0)
    settings = get_or_set_default(settings, 'max_cplex_time_per_iteration', 60.0)

    rho_idx = indices["rho"]
    loss_idx = indices["loss"]
    alpha_idx = indices["alpha"]
    cut_idx = loss_idx + rho_idx
    objval_idx = indices["objval"]
    L0_idx = indices["L0_norm"]

    if len(alpha_idx) == 0:
        get_alpha = lambda: np.array([])
    else:
        get_alpha = lambda: np.array(MIP.solution.get_values(alpha_idx))

    if type(loss_idx) is list and len(loss_idx) == 1:
        loss_idx = loss_idx[0]

    C_0_alpha = np.array(indices['C_0_alpha'])
    nnz_ind = np.flatnonzero(C_0_alpha)
    C_0_nnz = C_0_alpha[nnz_ind]

    if settings['update_bounds']:

        bounds = {
            'loss_min': MIP.variables.get_lower_bounds(loss_idx),
            'loss_max': MIP.variables.get_upper_bounds(loss_idx),
            'objval_min': MIP.variables.get_lower_bounds(objval_idx),
            'objval_max': MIP.variables.get_upper_bounds(objval_idx),
            'L0_min': MIP.variables.get_lower_bounds(L0_idx),
            'L0_max': MIP.variables.get_upper_bounds(L0_idx),
            }

        if settings['type'] == 'cvx':
            vtypes = 'C'
            update_problem_bounds = lambda bounds, lb, ub: chained_updates_for_lp(bounds, C_0_nnz, ub, lb)
        elif settings['type'] is 'ntree':
            vtypes = MIP.variables.get_types(rho_idx)
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
        MIP.parameters.timelimit.set(float(current_timelimit))
        MIP.solve()
        solution_status = MIP.solution.status[MIP.solution.get_status()]

        # get solution
        if solution_status in ('optimal', 'optimal_tolerance', 'MIP_optimal'):
            rho = np.array(MIP.solution.get_values(rho_idx))
            alpha = get_alpha()
            simplex_iteration = int(MIP.solution.progress.get_num_iterations())
        else:
            stop_reason = solution_status
            print_from_function('BREAKING NTREE CP LOOP NON-OPTIMAL SOLUTION FOUND: %s' % solution_status)
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
        lowerbound = MIP.solution.get_objective_value()
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
            print_update(rho, n_iterations, upperbound, lowerbound, relative_gap, vtypes)

        #save progress
        if settings['save_progress']:
            progress_stats['upperbounds'].append(upperbound)
            progress_stats['lowerbounds'].append(lowerbound)
            progress_stats['total_times'].append(total_time)
            progress_stats['cut_times'].append(cut_time)
            progress_stats['simplex_iterations'].append(simplex_iteration)

        # check termination conditions
        if relative_gap < settings['max_tolerance']:
            stop_reason = 'converged:gap_within_tolerance'
            print_from_function('BREAKING NTREE CP LOOP - MAX TOLERANCE')
            break

        if cplex_time > settings['max_cplex_time_per_iteration']:
            stop_reason = 'aborted:reached_max_train_time'
            break

        if iteration_time > settings['max_runtime_per_iteration']:
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('BREAKING NTREE CP LOOP - REACHED MAX RUNTIME PER ITERATION')
            break

        if (total_time > settings['max_runtime']) or (remaining_total_time == 0.0):
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('BREAKING NTREE CP LOOP - REACHED MAX RUNTIME')
            break

        # switch bounds
        if settings['update_bounds']:
            MIP.variables.set_lower_bounds(L0_idx, bounds['L0_min'])
            MIP.variables.set_upper_bounds(L0_idx, bounds['L0_max'])
            MIP.variables.set_lower_bounds(loss_idx, bounds['loss_min'])
            MIP.variables.set_upper_bounds(loss_idx, bounds['loss_max'])
            MIP.variables.set_lower_bounds(objval_idx, bounds['objval_min'])
            MIP.variables.set_upper_bounds(objval_idx, bounds['objval_max'])

        # add loss cut
        MIP.linear_constraints.add(lin_expr = cut_constraint,
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
    idx = range(indices['n_constraints'], MIP.linear_constraints.get_num(), 1)
    cuts = {
        'coefs': MIP.linear_constraints.get_rows(idx),
        'lhs': MIP.linear_constraints.get_rhs(idx)
    }

    #create solution pool
    if settings['type'] == 'ntree':
        C_0_rho = np.array(indices['C_0_rho'])
        for i in range(MIP.solution.pool.get_num()):
            rho = np.array(MIP.solution.pool.get_values(i, rho_idx))
            objval = compute_loss(rho) + np.sum(C_0_rho[np.flatnonzero(np.array(rho))])
            solutions.append(rho)
            objvals.append(objval)

    if len(objvals) > 1:
        pool = SolutionPool({'objvals': objvals, 'solutions': solutions})
    else:
        pool = SolutionPool(len(indices["rho"]))

    return stats, cuts, pool
