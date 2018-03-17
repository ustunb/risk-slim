import numpy as np
import time
from .mip import create_risk_slim, set_cplex_mip_parameters
from .lattice_cpa import setup_penalty_parameters
from .standard_cpa import run_standard_cpa
from .heuristics import sequential_rounding, discrete_descent
from .bound_tightening import chained_updates
from .solution_classes import SolutionPool
from .debug import ipsh


INITIALIZATION_SETTINGS = {
    'display_progress': True,  # print progress of initialization procedure
    'display_cplex_progress': False,  # print of CPLEX during intialization procedure
    'max_runtime': 300.0,  # max time to run CPA in initialization procedure
    'max_iterations': 10000,  # max # of cuts needed to stop CPA
    'max_tolerance': 0.0001,  # tolerance of solution to stop CPA
    'max_runtime_per_iteration': 300.0,  # max time per iteration of CPA
    'max_cplex_time_per_iteration': 10.0,  # max time per iteration to solve surrogate problem in CPA
    'use_sequential_rounding': True,  # use SeqRd in initialization procedure
    'sequential_rounding_max_runtime': 30.0,  # max runtime for SeqRd in initialization procedure
    'sequential_rounding_max_solutions': 5,  # max solutions to round using SeqRd
    'polishing_after': True,  # polish after rounding
    'polishing_max_runtime': 30.0,  # max runtime for polishing
    'polishing_max_solutions': 5,  # max solutions to polish
    }


# Initialization Procedure
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
    cpa_stats, cuts, pool = run_standard_cpa(risk_slim_lp, risk_slim_lp_indices, settings, compute_loss_real, compute_loss_cut_real)

    # update bounds
    bounds = chained_updates(bounds, C_0_nnz, new_objval_at_relaxation = cpa_stats['lowerbound'])

    # remove redundant solutions, remove infeasible solutions, order solutions by objective value of RiskSLIMLP
    if settings['use_sequential_rounding']:

        pool, _, _ = sequential_round_solution_pool(pool = pool,
                                                    Z = Z,
                                                    C_0 = C_0,
                                                    compute_loss_from_scores_real = compute_loss_from_scores_real,
                                                    get_L0_penalty = get_L0_penalty,
                                                    max_runtime = settings['sequential_rounding_max_runtime'],
                                                    max_solutions = settings['sequential_rounding_max_solutions'],
                                                    objval_cutoff = bounds['objval_max'],
                                                    L0_min = bounds['L0_min'],
                                                    L0_max = bounds['L0_max'])

        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))

    else:
        pool, _, _ = round_solution_pool(pool, constraints)
        pool = pool.compute_objvals(get_objval)

    if settings['polishing_after'] and len(pool) > 0:
        pool, _, _ = discrete_descent_solution_pool(pool = pool,
                                                    Z = Z,
                                                    C_0 = C_0,
                                                    constraints = constraints,
                                                    compute_loss_from_scores = compute_loss_from_scores,
                                                    get_L0_penalty = get_L0_penalty,
                                                    max_runtime = settings['polishing_max_runtime'],
                                                    max_solutions = settings['polishing_max_solutions'])

        pool = pool.remove_infeasible(is_feasible).distinct().sort()

    if len(pool) > 0:
        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))

    return pool, cuts, bounds


def round_solution_pool(pool, constraints):
    """

    Parameters
    ----------
    pool
    constraints

    Returns
    -------

    """

    pool = pool.distinct().sort()
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
            pool = pool.remove_infeasible(rounded_model_size_is_ok)

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
    rounded_pool = rounded_pool.distinct().sort()
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


