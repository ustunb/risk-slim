import numpy as np
import time
from .mip import create_risk_slim, set_cplex_mip_parameters
from .lattice_cpa import setup_penalty_parameters
from .standard_cpa import run_standard_cpa
from .heuristics import sequential_rounding, discrete_descent
from .bound_tightening import chained_updates
from .solution_classes import SolutionPool

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

    # trade-off parameter
    _, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(c0_value = c0_value, coef_set = constraints['coef_set'])

    risk_slim_settings = dict(risk_slim_settings)
    risk_slim_settings.update(bounds)

    settings = dict(settings)
    settings['type'] = 'cvx'

    #create RiskSLIM LP
    risk_slim_settings['relax_integer_variables'] = True
    risk_slim_lp, risk_slim_lp_indices = create_risk_slim(risk_slim_settings)
    risk_slim_lp = set_cplex_mip_parameters(risk_slim_lp, cplex_settings, display_cplex_progress = settings['display_cplex_progress'])

    # solve risk_slim_lp LP using standard CPA
    cpa_stats, cuts, cpa_pool = run_standard_cpa(cpx = risk_slim_lp,
                                                 cpx_indices = risk_slim_lp_indices,
                                                 compute_loss = compute_loss_real,
                                                 compute_loss_cut = compute_loss_cut_real,
                                                 settings = settings)

    # update bounds
    bounds = chained_updates(bounds, C_0_nnz, new_objval_at_relaxation = cpa_stats['lowerbound'])

    # build bool of feasible solutions by rounding and polishing cpa
    pool = SolutionPool(cpa_pool.P)

    # apply rounding to CPA solutions
    if settings['use_rounding']:
        rnd_pool, _, _ = round_solution_pool(cpa_pool, constraints,
                                             max_runtime = settings['rounding_max_runtime'],
                                             max_solutions = settings['rounding_max_solutions'])
        rnd_pool = rnd_pool.compute_objvals(get_objval)
        pool.append(rnd_pool)

    # apply sequential rounding to CPA solutions
    if settings['use_sequential_rounding']:

        sqrnd_pool, _, _ = sequential_round_solution_pool(pool = cpa_pool,
                                                          Z = Z,
                                                          C_0 = C_0,
                                                          compute_loss_from_scores_real = compute_loss_from_scores_real,
                                                          get_L0_penalty = get_L0_penalty,
                                                          max_runtime = settings['sequential_rounding_max_runtime'],
                                                          max_solutions = settings['sequential_rounding_max_solutions'],
                                                          objval_cutoff = bounds['objval_max'],
                                                          L0_min = bounds['L0_min'],
                                                          L0_max = bounds['L0_max'])

        pool.append(sqrnd_pool)

    # apply polishing to rounded solutions
    if len(pool) > 0 and settings['polishing_after']:

        dcd_pool, _, _ = discrete_descent_solution_pool(pool = pool,
                                                        Z = Z,
                                                        C_0 = C_0,
                                                        constraints = constraints,
                                                        compute_loss_from_scores = compute_loss_from_scores,
                                                        get_L0_penalty = get_L0_penalty,
                                                        max_runtime = settings['polishing_max_runtime'],
                                                        max_solutions = settings['polishing_max_solutions'])

        pool.append(dcd_pool)

    # remove solutions that are not feasible, not integer
    if len(pool) > 0:
        pool = pool.remove_nonintegral().remove_infeasible(is_feasible).distinct().sort()

    # update upper and lower bounds
    if len(pool) > 0:
        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))

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
    L0_reg_ind = np.isnan(constraints['coef_set'].C_0j)
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
                                   objval_cutoff = float('inf'),
                                   L0_min = 0,
                                   L0_max = float('inf')):

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
    if L0_min > 0 and L0_max < pool.P:

        def rounded_model_size_is_ok(rho):
            penalized_rho = abs(rho)[C_0 == 0.0]
            rounded_rho_l0_min = np.count_nonzero(np.floor(penalized_rho))
            rounded_rho_l0_max = np.count_nonzero(np.ceil(penalized_rho))
            return rounded_rho_l0_max >= L0_min and rounded_rho_l0_min <= L0_max

        pool = pool.remove_infeasible(rounded_model_size_is_ok)
        pool = pool.distinct().sort()

        if len(pool) == 0:
            return pool, 0.0, 0


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


