import os
import numpy as np
from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.CoefficientSet import CoefficientSet
from riskslim.lattice_cpa import get_conservative_offset, run_lattice_cpa

#double check BLAS configuration
np.__config__.show()

# data
data_name = "breastcancer"                                  # name of the data
data_dir = os.getcwd() + '/datasets/'                       # directory where datasets are stored
data_csv_file = data_dir + data_name + '_processed.csv'     # csv file for the dataset
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
coef_set = CoefficientSet(variable_names=data['variable_names'], lb=-max_coefficient, ub=max_coefficient, sign=0)
coef_set.view()

# offset value
conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
max_offset = min(max_offset, conservative_offset)
coef_set.set_field('lb', '(Intercept)', -max_offset)
coef_set.set_field('ub', '(Intercept)', max_offset)
coef_set.view()

# create constraint dictionary
trivial_L0_max = P - np.sum(coef_set.C_0j == 0)
max_L0_value = min(max_L0_value, trivial_L0_max)

constraints = {
    'L0_min': 0,
    'L0_max': max_L0_value,
    'coef_set':coef_set,
}

# Run RiskSLIM
settings = {
    #
    'c0_value': c0_value,
    'w_pos': w_pos,
    #
    'max_runtime': 300.0,
    'max_tolerance': 0.000001,
    'max_iterations': 100000,
    'display_cplex_progress': True,
    #
    'loss_computation': 'normal',
    'update_bounds_flag': True,                         # use chained updates
    'polish_flag': True,                                # polish integer feasible solutions with DCD
    'round_flag': False,                                # round continuous solutions with SeqRd
    'polish_rounded_solutions': True,                   # polish solutions rounded with SeqRd using DCD
    'tight_formulation': True,                          # use tighter MIP formulation (forces alpha_j = 0 when lambda_j = 0)
    'add_cuts_at_heuristic_solutions': True,
    'initialization_flag': False,
    #
    'polishing_ub_to_objval_relgap': 0.1,
    'polishing_max_runtime': 10.0,
    'polishing_max_solutions': 5.0,
    'polishing_min_cuts': 0,
    'polishing_max_cuts': float('inf'),
    'polishing_min_relgap': 5.0,
    'polishing_max_relgap': float('inf'),
    #
    'rounding_min_cuts': 0,
    'rounding_max_cuts': 20000,
    'rounding_min_relgap': 0.2,
    'rounding_max_relgap': float('inf'),
    'rounding_ub_to_objval_relgap': float('inf'),
    #
    'init_display_progress': True,
    'init_display_cplex_progress': False,
    #
    'init_max_runtime': 300.0,
    'init_max_runtime_per_iteration': 300.0,
    'init_max_cplex_time_per_iteration': 10.0,
    'init_max_iterations': 10000,
    'init_max_tolerance': 0.0001,
    #
    'init_use_sequential_rounding': True,
    'init_sequential_rounding_max_runtime': 30.0,
    'init_sequential_rounding_max_solutions': 5,
    'init_polishing_after': True,
    'init_polishing_max_runtime': 30.0,
    'init_polishing_max_solutions': 5,
    #
    'cplex_randomseed': 0,
    'cplex_mipemphasis': 0,
    'cplex_mipgap': np.finfo('float').eps,
    'cplex_absmipgap': np.finfo('float').eps,
    'cplex_integrality_tolerance': np.finfo('float').eps,
    'cplex_repairtries': 20,
    'cplex_poolsize': 100,
    'cplex_poolrelgap': float('nan'),
    'cplex_poolreplace': 2,
    'cplex_n_cores': 1,
    'cplex_nodefilesize': (120 * 1024) / 1,
}

# train model using lattice_cpa
output = run_lattice_cpa(data, constraints, settings)

# print output
pprint(output)

# just a placeholder for now
print(print_model(output['incumbent'], data))


