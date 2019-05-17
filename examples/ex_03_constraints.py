import os
import numpy as np
import cplex as cplex
from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa

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

# load data from disk
data = load_data_from_csv(dataset_csv_file = data_csv_file, sample_weights_csv_file = sample_weights_csv_file)
N, P = data['X'].shape

# create coefficient set and set the value of the offset parameter
coef_set = CoefficientSet(variable_names=data['variable_names'], lb=-max_coefficient, ub=max_coefficient, sign=0)
conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
max_offset = min(max_offset, conservative_offset)
coef_set['(Intercept)'].ub = max_offset
coef_set['(Intercept)'].lb = -max_offset

# create constraint
trivial_L0_max = P - np.sum(coef_set.C_0j == 0)
max_L0_value = min(max_L0_value, trivial_L0_max)

constraints = {
    'L0_min': 0,
    'L0_max': max_L0_value,
    'coef_set':coef_set,
}


# major settings (see riskslim_ex_02_complete for full set of options)
settings = {
    # Problem Parameters
    'c0_value': c0_value,
    'w_pos': w_pos,
    #
    # LCPA Settings
    'max_runtime': 300.0,                               # max runtime for LCPA
    'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    'display_cplex_progress': True,                     # print CPLEX progress on screen
    'loss_computation': 'normal',                       # how to compute the loss function ('normal','fast','lookup')
    #
    # RiskSLIM MIP settings
    'drop_variables': False,
    #
    # LCPA Improvements
    'round_flag': False,                                # round continuous solutions with SeqRd
    'polish_flag': False,                               # polish integer feasible solutions with DCD
    'chained_updates_flag': False,                      # use chained updates
    'initialization_flag': False,                       # use initialization procedure
    'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
    'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
    #
    # CPLEX Solver Parameters
    'cplex_randomseed': 0,                              # random seed
    'cplex_mipemphasis': 0,                             # cplex MIP strategy
}

# turn on at your own risk
settings['round_flag'] = False
settings['polish_flag'] = False
settings['chained_updates_flag'] = False
settings['initialization_flag'] = False


# initialize MIP for lattice CPA
mip_objects = setup_lattice_cpa(data, constraints, settings)

# add operational constraints
mip, indices = mip_objects['mip'], mip_objects['indices']
get_alpha_name = lambda var_name: 'alpha_' + str(data['variable_names'].index(var_name))
get_alpha_ind = lambda var_names: [get_alpha_name(v) for v in var_names]

# to add a constraint like "either "CellSize" or "CellShape"
# you must formulate the constraint in terms of the alpha variables
# alpha[cell_size] + alpha[cell_shape] <= 1 to MIP
mip.linear_constraints.add(
        names = ["EitherOr_CellSize_or_CellShape"],
        lin_expr = [cplex.SparsePair(ind = get_alpha_ind(['UniformityOfCellSize', 'UniformityOfCellShape']),
                                     val = [1.0, 1.0])],
        senses = "L",
        rhs = [1.0])

mip_objects['mip'] = mip

# pass MIP back to lattice CPA so that it will solve
model_info, mip_info, lcpa_info = finish_lattice_cpa(data, constraints, mip_objects, settings)

#model info contains key results
pprint(model_info)
print_model(model_info['solution'], data)

# mip_output contains information to access the MIP
mip_info['risk_slim_mip'] #CPLEX mip
mip_info['risk_slim_idx'] #indices of the relevant constraints

# lcpa_output contains detailed information about LCPA
pprint(lcpa_info)




