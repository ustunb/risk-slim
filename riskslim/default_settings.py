import numpy as np

# Lattice CPA
DEFAULT_LCPA_SETTINGS = {
    #
    'c0_value': 1e-6,
    'w_pos': 1.00,
    'tight_formulation': True,     #use a slightly tighter MIP formulation
    'drop_variables': True,        #drop variables
    #
    # LCPA Settings
    'max_runtime': 300.0,  # max runtime for LCPA
    'max_tolerance': 0.000001,  # tolerance to stop LCPA
    'display_cplex_progress': True,  # setting to True shows CPLEX progress
    'loss_computation': 'normal',  # type of loss computation to use ('normal','fast','lookup')
    'chained_updates_flag': True,  # use chained updates
    'initialization_flag': False,  # use initialization procedure
    'add_cuts_at_heuristic_solutions': True, #add cuts at integer feasible solutions found using polishing/rounding
    #
    #  LCPA Rounding Heuristic
    'round_flag': True,  # round continuous solutions with SeqRd
    'polish_rounded_solutions': True,  # polish solutions rounded with SeqRd using DCD
    'rounding_tolerance': float('inf'),  # only solutions with objective value < (1 + tol) are rounded
    'rounding_start_cuts': 0,  # cuts needed to start using rounding heuristic
    'rounding_start_gap': float('inf'),  # optimality gap needed to start using rounding heuristic
    'rounding_stop_cuts': 20000,  # cuts needed to stop using rounding heuristic
    'rounding_stop_gap': 0.2,  # optimality gap needed to stop using rounding heuristic
    #
    # LCPA Polishing Heuristic
    'polish_flag': True,  # polish integer feasible solutions with DCD
    'polishing_tolerance': 0.1, # only solutions with objective value (1 + polishing_ub_to_objval_relgap) are polished. setting to
    'polishing_max_runtime': 10.0,  # max time to run polishing each time
    'polishing_max_solutions': 5.0,  # max # of solutions to polish each time
    'polishing_start_cuts': 0,  # cuts needed to start using polishing heuristic
    'polishing_start_gap': float('inf'),  # min optimality gap needed to start using polishing heuristic
    'polishing_stop_cuts': float('inf'),  # cuts needed to stop using polishing heuristic
    'polishing_stop_gap': 5.0,  # max optimality gap required to stop using polishing heuristic
    #
    # Initialization Procedure
    'init_display_progress': True,  # print progress of initialization procedure
    'init_display_cplex_progress': False,  # print of CPLEX during intialization procedure
    'init_max_runtime': 300.0,  # max time to run CPA in initialization procedure
    'init_max_iterations': 10000,  # max # of cuts needed to stop CPA
    'init_max_tolerance': 0.0001,  # tolerance of solution to stop CPA
    'init_max_rounding_gap': 0.49,
    'init_max_runtime_per_iteration': 300.0,  # max time per iteration of CPA
    'init_max_cplex_time_per_iteration': 10.0,  # max time per iteration to solve surrogate problem in CPA
    #
    'init_use_rounding': True,  # use Rd in initialization procedure
    'init_rounding_max_solutions': 100, # max runtime for Rd in initialization procedure
    'init_rounding_max_runtime': 20.0, # max runtime for Rd in initialization procedure
    #
    'init_use_sequential_rounding': False,  # use SeqRd in initialization procedure
    'init_sequential_rounding_max_runtime': 30.0,  # max runtime for SeqRd in initialization procedure
    'init_sequential_rounding_max_solutions': 5,  # max solutions to round using SeqRd
    #
    'init_polishing_after': True,  # polish after rounding
    'init_polishing_max_runtime': 30.0,  # max runtime for polishing
    'init_polishing_max_solutions': 5,  # max solutions to polish
    #
    #  CPLEX Solver Parameters
    'cplex_randomseed': 0,  # random seed
    'cplex_mipemphasis': 0,  # cplex MIP strategy
    'cplex_mipgap': np.finfo('float').eps,  #
    'cplex_absmipgap': np.finfo('float').eps,  #
    'cplex_integrality_tolerance': np.finfo('float').eps,  #
    'cplex_repairtries': 20,  # number of tries to repair user provided solutions
    'cplex_poolsize': 100,  # number of feasible solutions to keep in solution pool
    'cplex_poolrelgap': float('nan'),  # discard if solutions
    'cplex_poolreplace': 2,  # solution pool
    'cplex_n_cores': 1,  # number of cores to use in B & B (must be 1)
    'cplex_nodefilesize': (120 * 1024) / 1,  # node file size
    #
    #  Internal Parameters
    'purge_loss_cuts': False,
    'purge_bound_cuts': False,
    }


DEFAULT_CPA_SETTINGS = {
    'display_progress': True,  # print progress of initialization procedure
    'display_cplex_progress': True,  # print of CPLEX during intialization procedure
    'max_coefficient_gap': 0.5,  # tolerance of solution to stop CPA
    'max_runtime': 300.0,  # max time to run CPA in initialization procedure
    'max_iterations': 10000,  # max # of cuts needed to stop CPA
    'max_tolerance': 0.0001,  # tolerance of solution to stop CPA
    'max_runtime_per_iteration': 300.0,  # max time per iteration of CPA
    'max_cplex_time_per_iteration': 10.0,  # max time per iteration to solve surrogate problem in CPA
    }

DEFAULT_INITIALIZATION_SETTINGS = {
    #
    'use_rounding': True,  # use SeqRd in initialization procedure
    'rounding_max_runtime': 30.0,  # max runtime for Rs in initialization procedure
    'rounding_max_solutions': 5,  # max solutions to round using Rd
    #
    'use_sequential_rounding': True,  # use SeqRd in initialization procedure
    'sequential_rounding_max_runtime': 30.0,  # max runtime for SeqRd in initialization procedure
    'sequential_rounding_max_solutions': 5,  # max solutions to round using SeqRd
    #
    'polishing_after': True,  # polish after rounding
    'polishing_max_runtime': 30.0,  # max runtime for polishing
    'polishing_max_solutions': 5,  # max solutions to polish
    }
