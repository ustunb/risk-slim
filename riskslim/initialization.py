import numpy as np
from .mip import create_risk_slim, set_cplex_mip_parameters
from .lattice_cpa import setup_penalty_parameters
from .standard_cpa import run_standard_cpa
from .heuristics import sequential_round_solution_pool, discrete_descent_solution_pool, round_solution_pool
from .bound_tightening import chained_updates
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
    pool = pool.distinct().removeInfeasible(is_feasible).sort()

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

        pool = pool.distinct().sort()
        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))

    else:
        pool, _, _ = round_solution_pool(pool, constraints)
        pool.computeObjvals(get_objval)

    if settings['polishing_after'] and len(pool) > 0:
        pool, _, _ = discrete_descent_solution_pool(pool = pool,
                                                    Z = Z,
                                                    C_0 = C_0,
                                                    constraints = constraints,
                                                    compute_loss_from_scores = compute_loss_from_scores,
                                                    get_L0_penalty = get_L0_penalty,
                                                    max_runtime = settings['polishing_max_runtime'],
                                                    max_solutions = settings['polishing_max_solutions'])

        pool = pool.removeInfeasible(is_feasible).distinct().sort()

    if len(pool) > 0:
        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))

    return pool, cuts, bounds

