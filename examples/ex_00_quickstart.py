"""
00. Quickstart
"""
# pip install -e .

from riskslim import RiskSLIMClassifier, load_data_from_csv
from riskslim.utils import open_file
from pathlib import Path

# Load Data
data_name = "breastcancer"
data = load_data_from_csv(dataset_csv_file = Path(f'examples/data/{data_name}_data.csv'))

# Initialize Model
rs = RiskSLIMClassifier(
        max_size = 5, # max model size (number of non-zero coefficients; default set as float(inf))
        max_coef = 5, # value of largest/smallest coefficient
        variable_names = data["variable_names"],
        outcome_name = data["outcome_name"],
        verbose = False
        )
# Fit
rs.fit(X = data["X"], y = data["y"])

# Show Scores
rs.scores

# Create Report
report_file = rs.create_report(file_name = 'report.html', show = True)
open_file(report_file)



# # Major settings
# settings = {
#     # LCPA Settings
#     # -------------
#     # max runtime for LCPA
#     "max_runtime": 30.0,
#     # tolerance to stop LCPA (set to 0 to return provably optimal solution)
#     "max_tolerance": np.finfo("float").eps,
#     # how to compute the loss function ("normal","fast","lookup")
#     "loss_computation": "fast",
#
#     # LCPA Improvements
#     # -----------------
#     # round continuous solutions with SeqRd
#     "round_flag": True,
#     # polish integer feasible solutions with DCD
#     "polish_flag": True,
#     # use chained updates
#     "chained_updates_flag": True,
#     # add cuts at integer feasible solutions found using polishing/rounding
#     "add_cuts_at_heuristic_solutions": True,
#
#     # Initialization
#     # --------------
#     # use initialization procedure
#     "initialization_flag": True,
#     # max time to run CPA in initialization procedure
#     "init_max_runtime": 120.0,
#     "init_max_coefficient_gap": 0.49,
#
#     # CPLEX Solver Parameters
#     # -----------------------
#     # random seed
#     "cplex_randomseed": 0,
#     # cplex MIP strategy
#     "cplex_mipemphasis": 0,
#     }

