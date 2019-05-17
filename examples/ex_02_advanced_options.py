import os
import numpy as np
from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa

# data
data_name = "breastcancer"                                  # name of the data
data_dir = os.getcwd() + '/examples/data/'                  # directory where datasets are stored
data_csv_file = data_dir + data_name + '_data.csv'          # csv file for the dataset
sample_weights_csv_file = None                              # csv file of sample weights for the dataset (optional)

# problem parameters
max_coefficient = 5                                         # value of largest/smallest coefficient
max_L0_value = 5                                            # maximum model size
max_offset = 50                                             # maximum value of offset parameter (optional)
c0_value = 1e-6                                             # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
w_pos = 1.00                                                # relative weight on examples with y = +1; w_neg = 1.00 (optional)

# load dataset
data = load_data_from_csv(dataset_csv_file = data_csv_file, sample_weights_csv_file = sample_weights_csv_file)
N, P = data['X'].shape

# coefficient set
coef_set = CoefficientSet(variable_names = data['variable_names'], lb=-max_coefficient, ub=max_coefficient, sign=0)

# offset value
conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
max_offset = min(max_offset, conservative_offset)
coef_set['(Intercept)'].ub = max_offset
coef_set['(Intercept)'].lb = -max_offset

# create constraint dictionary
N, P = data['X'].shape
trivial_L0_max = P - np.sum(coef_set.C_0j == 0)
max_L0_value = min(max_L0_value, trivial_L0_max)

constraints = {
    'L0_min': 0,
    'L0_max': max_L0_value,
    'coef_set': coef_set,
}

# Run RiskSLIM
settings = {
    #
    'c0_value': c0_value,
    'w_pos': w_pos,
    #
    # LCPA Settings
    'max_runtime': 300.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # set to True to print CPLEX progress
    'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')
    #
    # Other LCPA Heuristics
    'chained_updates_flag': True,                         # use chained updates
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
    #
    # LCPA Rounding Heuristic
    'round_flag': False,                                 # round continuous solutions with SeqRd
    'polish_rounded_solutions': True,                   # polish solutions rounded with SeqRd using DCD
    'rounding_tolerance': float('inf'),                 # only solutions with objective value < (1 + tol) are rounded
    'rounding_start_cuts': 0,                           # cuts needed to start using rounding heuristic
    'rounding_start_gap': float('inf'),                 # optimality gap needed to start using rounding heuristic
    'rounding_stop_cuts': 20000,                        # cuts needed to stop using rounding heuristic
    'rounding_stop_gap': 0.2,                           # optimality gap needed to stop using rounding heuristic
    #
    # LCPA Polishing Heuristic
    'polish_flag': False,                                # polish integer feasible solutions with DCD
    'polishing_tolerance': 0.1,                         # only solutions with objective value (1 + tol) are polished.
    'polishing_max_runtime': 10.0,                      # max time to run polishing each time
    'polishing_max_solutions': 5.0,                     # max # of solutions to polish each time
    'polishing_start_cuts': 0,                          # cuts needed to start using polishing heuristic
    'polishing_start_gap': float('inf'),                # min optimality gap needed to start using polishing heuristic
    'polishing_stop_cuts': float('inf'),                # cuts needed to stop using polishing heuristic
    'polishing_stop_gap': 0.0,                          # max optimality gap required to stop using polishing heuristic
    #
    # Initialization Procedure
    'initialization_flag': True,                       # use initialization procedure
    'init_display_progress': True,                      # show progress of initialization procedure
    'init_display_cplex_progress': False,               # show progress of CPLEX during intialization procedure
    #
    'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
    'init_max_iterations': 10000,                       # max # of cuts needed to stop CPA
    'init_max_tolerance': 0.0001,                       # tolerance of solution to stop CPA
    'init_max_runtime_per_iteration': 300.0,            # max time per iteration of CPA
    'init_max_cplex_time_per_iteration': 10.0,          # max time per iteration to solve surrogate problem in CPA
    #
    'init_use_rounding': True,                          # use Rd in initialization procedure
    'init_rounding_max_runtime': 30.0,                  # max runtime for Rd in initialization procedure
    'init_rounding_max_solutions': 5,                   # max solutions to round using Rd
    #
    'init_use_sequential_rounding': True,               # use SeqRd in initialization procedure
    'init_sequential_rounding_max_runtime': 10.0,       # max runtime for SeqRd in initialization procedure
    'init_sequential_rounding_max_solutions': 5,        # max solutions to round using SeqRd
    #
    'init_polishing_after': True,                       # polish after rounding
    'init_polishing_max_runtime': 30.0,                 # max runtime for polishing
    'init_polishing_max_solutions': 5,                  # max solutions to polish
    #
    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}

# train model using lattice_cpa
model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)

#model info contains key results
pprint(model_info)
print_model(model_info['solution'], data)

# mip_output contains information to access the MIP
mip_info['risk_slim_mip'] #CPLEX mip
mip_info['risk_slim_idx'] #indices of the relevant constraints

# lcpa_output contains detailed information about LCPA
pprint(lcpa_info)





