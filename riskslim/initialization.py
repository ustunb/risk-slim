import time
import numpy as np
from .bound_tightening import chained_updates
from .helper_functions import print_log
from .heuristics import discrete_descent, sequential_rounding
from .setup_functions import setup_penalty_parameters
from .mip import create_risk_slim, set_cplex_mip_parameters
from .solution_classes import SolutionPool
from .standard_cpa import run_standard_cpa


def initialize_lattice_cpa(Z,
                           c0_value,
                           constraints,
                           bounds,
                           settings,
                           risk_slim_settings,
                           cplex_settings,
                           compute_loss_real,
                           compute_loss_cut_real,
                           compute_loss_from_scores_real,
                           compute_loss_from_scores,
                           get_objval,
                           get_L0_penalty,
                           is_feasible):
    """

    Returns
    -------
    cuts
    solution pool
    bounds

    """
    #To-Do recompute function handles here if required
    assert callable(compute_loss_real)
    assert callable(compute_loss_cut_real)
    assert callable(compute_loss_from_scores_real)
    assert callable(compute_loss_from_scores)
    assert callable(get_objval)
    assert callable(get_L0_penalty)
    assert callable(is_feasible)

    print_log('-' * 60)
    print_log('runnning initialization procedure')
    print_log('-' * 60)

    # trade-off parameter
    _, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(c0_value = c0_value, coef_set = constraints['coef_set'])


    settings = dict(settings)
    settings['type'] = 'cvx'

    #create RiskSLIM LP
    risk_slim_settings = dict(risk_slim_settings)
    risk_slim_settings.update(bounds)
    risk_slim_settings['relax_integer_variables'] = True
    risk_slim_lp, risk_slim_lp_indices = create_risk_slim(coef_set = constraints['coef_set'],
                                                          input = risk_slim_settings)

    risk_slim_lp = set_cplex_mip_parameters(risk_slim_lp, cplex_settings, display_cplex_progress = settings['display_cplex_progress'])

    # solve risk_slim_lp LP using standard CPA
    cpa_stats, cuts, cpa_pool = run_standard_cpa(cpx = risk_slim_lp,
                                                 cpx_indices = risk_slim_lp_indices,
                                                 compute_loss = compute_loss_real,
                                                 compute_loss_cut = compute_loss_cut_real,
                                                 settings = settings)

    # update bounds
    bounds = chained_updates(bounds, C_0_nnz, new_objval_at_relaxation = cpa_stats['lowerbound'])

    def rounded_model_size_is_ok(rho):
        zero_idx_rho_ceil = np.equal(np.ceil(rho), 0)
        zero_idx_rho_floor = np.equal(np.floor(rho), 0)
        cannot_round_to_zero = np.logical_not(np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor))
        rounded_rho_L0_min = np.count_nonzero(cannot_round_to_zero[L0_reg_ind])
        rounded_rho_L0_max = np.count_nonzero(rho[L0_reg_ind])
        return rounded_rho_L0_min >= constraints['L0_min'] and rounded_rho_L0_max <= constraints['L0_max']

    cpa_pool = cpa_pool.remove_infeasible(rounded_model_size_is_ok).distinct().sort()

    if len(cpa_pool) == 0:
        print_log('all solutions from CPA are infeasible')

    # build bool of feasible solutions by rounding and polishing cpa
    pool = SolutionPool(cpa_pool.P)

    # apply rounding to CPA solutions
    if settings['use_rounding'] and len(cpa_pool) > 0:
        print_log('running normal rounding on %d solutions' % len(cpa_pool))
        print_log('best objective value is %1.4f' %np.min(cpa_pool.objvals))
        rnd_pool, _, _ = round_solution_pool(cpa_pool,
                                             constraints,
                                             max_runtime = settings['rounding_max_runtime'],
                                             max_solutions = settings['rounding_max_solutions'])
        rnd_pool = rnd_pool.compute_objvals(get_objval)

        if len(rnd_pool) > 0:
            print_log('rounding produced %d solutions' % len(rnd_pool))
            print_log('best objective value is %1.4f' % np.min(rnd_pool.objvals))
            pool.append(rnd_pool)
        else:
            print_log('all solutions produced by rounding were infeasible')

    # apply sequential rounding to CPA solutions
    if settings['use_sequential_rounding'] and len(cpa_pool) > 0:
        print_log('running sequential rounding on %d solutions' % len(cpa_pool))
        print_log('best objective value is %1.4f' % np.min(cpa_pool.objvals))
        sqrnd_pool, _, _ = sequential_round_solution_pool(pool = cpa_pool,
                                                          Z = Z,
                                                          C_0 = C_0,
                                                          compute_loss_from_scores_real = compute_loss_from_scores_real,
                                                          get_L0_penalty = get_L0_penalty,
                                                          max_runtime = settings['sequential_rounding_max_runtime'],
                                                          max_solutions = settings['sequential_rounding_max_solutions'],
                                                          objval_cutoff = bounds['objval_max'])

        if len(sqrnd_pool) > 0:
            print_log('sequential rounding produced %d solutions' % len(sqrnd_pool))
            print_log('best objective value is %1.4f' % np.min(sqrnd_pool.objvals))
            pool.append(sqrnd_pool)
        else:
            print_log('all solutions produced by rounding were infeasible')

    pool = pool.remove_infeasible(is_feasible)
    if len(pool) == 0:
        print_log('all rounded solutions from CPA are infeasible')

    # apply polishing to rounded solutions
    if settings['polishing_after'] and len(pool) > 0:
        print_log('running polishing on %d solutions' % len(pool))
        print_log('best objective value is %1.4f' % np.min(pool.objvals))
        dcd_pool, _, _ = discrete_descent_solution_pool(pool = pool,
                                                        Z = Z,
                                                        C_0 = C_0,
                                                        constraints = constraints,
                                                        compute_loss_from_scores = compute_loss_from_scores,
                                                        get_L0_penalty = get_L0_penalty,
                                                        max_runtime = settings['polishing_max_runtime'],
                                                        max_solutions = settings['polishing_max_solutions'])

        dcd_pool = dcd_pool.remove_infeasible(is_feasible)
        if len(dcd_pool) > 0:
            print_log('polishing produced %d feasible solutions' % len(dcd_pool))
            print_log('best objective value is %1.4f' % np.min(dcd_pool.objvals))
            pool.append(dcd_pool)
        else:
            print_log('all solutions produced by polishing were infeasible')

    # remove solutions that are not feasible, not integer
    if len(pool) > 0:
        pool = pool.remove_nonintegral().remove_infeasible(is_feasible).distinct().sort()
        print_log('initial pool statistics')
        print_log('number of feasible solutions: %d' % len(pool))
        print_log('objective value of best solution: %1.4f' % np.min(pool.objvals))

    # update upper and lower bounds
    if len(pool) > 0:
        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))

    print_log('-' * 60)
    print_log('ran initialization procedure')
    print_log('-' * 60)
    return pool, cuts, bounds


def round_solution_pool(pool,
                        constraints,
                        max_runtime = float('inf'),
                        max_solutions = float('inf')):
    """

    Parameters
    ----------
    pool
    constraints
    max_runtime
    max_solutions

    Returns
    -------

    """
    # quick return
    if len(pool) == 0:
        return pool

    pool = pool.distinct().sort()
    P = pool.P
    L0_reg_ind = np.isnan(constraints['coef_set'].c0)
    L0_max = constraints['L0_max']


    total_runtime = 0.0
    total_rounded = 0
    rounded_pool = SolutionPool(P)

    for rho in pool.solutions:

        start_time = time.time()
        # sort from largest to smallest coefficients
        feature_order = np.argsort([-abs(x) for x in rho])
        rounded_solution = np.zeros(shape = (1, P))
        l0_norm_count = 0

        for k in range(P):
            j = feature_order[k]
            if not L0_reg_ind[j]:
                rounded_solution[0, j] = np.round(rho[j], 0)
            elif l0_norm_count < L0_max:
                rounded_solution[0, j] = np.round(rho[j], 0)
                l0_norm_count += L0_reg_ind[j]

        total_runtime += time.time() - start_time
        total_rounded += 1
        rounded_pool.add(objvals = np.nan, solutions = rounded_solution)

        if total_runtime > max_runtime or total_rounded >= max_solutions:
            break

    rounded_pool = rounded_pool.distinct().sort()
    return rounded_pool, total_runtime, total_rounded


def sequential_round_solution_pool(pool,
                                   Z,
                                   C_0,
                                   compute_loss_from_scores_real,
                                   get_L0_penalty,
                                   max_runtime = float('inf'),
                                   max_solutions = float('inf'),
                                   objval_cutoff = float('inf')):

    """
    runs sequential rounding for all solutions in a solution pool
    can be stopped early using max_runtime or max_solutions

    Parameters
    ----------
    pool
    Z
    C_0
    compute_loss_from_scores_real
    get_L0_penalty
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

    assert callable(get_L0_penalty)
    assert callable(compute_loss_from_scores_real)

    # if model size constraint is non-trivial, remove solutions that violate the model size constraint beforehand
    pool = pool.distinct().sort()
    rounding_handle = lambda rho: sequential_rounding(rho = rho,
                                                      Z = Z,
                                                      C_0 = C_0,
                                                      compute_loss_from_scores_real = compute_loss_from_scores_real,
                                                      get_L0_penalty = get_L0_penalty,
                                                      objval_cutoff = objval_cutoff)


    # apply sequential rounding to all solutions
    total_runtime = 0.0
    total_rounded = 0
    rounded_pool = SolutionPool(pool.P)

    for rho in pool.solutions:

        start_time = time.time()
        solution, objval, early_stop = rounding_handle(rho)
        total_runtime += time.time() - start_time
        total_rounded += 1

        if not early_stop:
            rounded_pool = rounded_pool.add(objvals = objval, solutions = solution)

        if total_runtime > max_runtime or total_rounded > max_solutions:
            break

    rounded_pool = rounded_pool.distinct().sort()
    return rounded_pool, total_runtime, total_rounded


def discrete_descent_solution_pool(pool,
                                   Z,
                                   C_0,
                                   constraints,
                                   get_L0_penalty,
                                   compute_loss_from_scores,
                                   max_runtime = float('inf'),
                                   max_solutions = float('inf')):
    """

    runs dcd polishing for all solutions in a solution pool
    can be stopped early using max_runtime or max_solutions


    Parameters
    ----------
    pool
    Z
    C_0
    constraints
    get_L0_penalty
    compute_loss_from_scores
    max_runtime
    max_solutions

    Returns
    -------

    """
    pool = pool.remove_nonintegral()

    # quick return
    if len(pool) == 0:
        return pool, 0.0, 0

    assert callable(get_L0_penalty)
    assert callable(compute_loss_from_scores)

    rho_ub = constraints['coef_set'].ub
    rho_lb = constraints['coef_set'].lb

    polishing_handle = lambda rho: discrete_descent(rho,
                                                    Z = Z,
                                                    C_0 = C_0,
                                                    rho_ub = rho_ub,
                                                    rho_lb = rho_lb,
                                                    get_L0_penalty = get_L0_penalty,
                                                    compute_loss_from_scores = compute_loss_from_scores)
    pool = pool.distinct().sort()
    polished_pool = SolutionPool(pool.P)
    total_runtime = 0.0
    total_polished = 0

    for rho in pool.solutions:

        start_time = time.time()
        polished_solution, _, polished_objval = polishing_handle(rho)
        total_runtime += time.time() - start_time
        total_polished += 1
        polished_pool = polished_pool.add(objvals = polished_objval, solutions = polished_solution)

        if total_runtime > max_runtime or total_polished >= max_solutions:
            break

    polished_pool = polished_pool.distinct().sort()
    return polished_pool, total_runtime, total_polished
