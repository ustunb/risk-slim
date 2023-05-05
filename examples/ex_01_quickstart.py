"""
01. Quickstart
==============

A minimial example for learning risk scores.
"""

###################################################################################################

import os
import numpy as np
from riskslim import RiskSLIMClassifier, load_data_from_csv

###################################################################################################
# Load Data
# ---------
#
# The example dataset in this tutorial is used to predict if a breast cancer tumor is beign or
# malignant.
#

# Paths
data_name = "breastcancer"                  # name of the data
data_dir = "../examples"                    # directory where datasets are stored
data_csv_file = os.path.join(               # csv file for the dataset
    data_dir, "data", data_name+"_data.csv"
)
sample_weights_csv_file = None              # csv file of sample weights for the dataset (optional)


# Load data
data = load_data_from_csv(
    dataset_csv_file=data_csv_file, sample_weights_csv_file=sample_weights_csv_file
)

###################################################################################################
# Settings
# --------
#
# RiskSLIM settings and brief descriptions are provided below. See the next example for
# a full set of options.
#

# Major settings
settings = {
    # LCPA Settings
    # -------------
    # max runtime for LCPA
    "max_runtime": 30.0,
    # tolerance to stop LCPA (set to 0 to return provably optimal solution)
    "max_tolerance": np.finfo("float").eps,
    # how to compute the loss function ("normal","fast","lookup")
    "loss_computation": "fast",

    # LCPA Improvements
    # -----------------
    # round continuous solutions with SeqRd
    "round_flag": True,
    # polish integer feasible solutions with DCD
    "polish_flag": True,
    # use chained updates
    "chained_updates_flag": True,
    # add cuts at integer feasible solutions found using polishing/rounding
    "add_cuts_at_heuristic_solutions": True,

    # Initialization
    # --------------
    # use initialization procedure
    "initialization_flag": True,
    # max time to run CPA in initialization procedure
    "init_max_runtime": 120.0,
    "init_max_coefficient_gap": 0.49,

    # CPLEX Solver Parameters
    # -----------------------
    # random seed
    "cplex_randomseed": 0,
    # cplex MIP strategy
    "cplex_mipemphasis": 0,
}


###################################################################################################
# Problem Parameters
# ------------------
#
# These parameters determine the sparisty and constraints of the model. The bounds on magnitiude of
# coefficients and number of non-zero coefficents are set below. Pass these parameters to a
# ``RiskSLIM`` object during initialization.
#

# Value of largest/smallest coefficient
max_coefficient = 5

# Maximum model size (number of non-zero coefficients; default set as float(inf))
max_L0 = 5

# Maximum value of offset (intercept) parameter (optional)
max_offset = 50

# L0-penalty parameter
#   c0_value > 0
#   larger values -> sparser models
#   small values (1e-6) give model with max_L0 terms
c0_value = 1e-6


###################################################################################################
# Initialize & Fit
# ----------------
#
# The ``RiskSLIM`` model is first initalized using the coefficient set, bounds on the number of
# non-zero coefficients (``L0_min`` and ``L0_max``), and the settings defined in the previous
# cell. Fitting the model object is performed using ``.fit(X, y)``, where ``X`` is a 2d-array of
# features and ``y`` is an array of class labels.
#

rs = RiskSLIMClassifier(
    L0_min=0,
    L0_max=max_L0,
    rho_max=max_coefficient,
    rho_min=-max_coefficient,
    c0_value=c0_value,
    max_abs_offset=max_offset,
    settings=settings,
    verbose=False
)

rs.fit(
    data["X"],
    data["y"],
    variable_names=data["variable_names"],
    outcome_name=data["outcome_name"]
)

###################################################################################################
# Results
# -------
#
# The CPLEX object and dictionary of solution is store in the ``.solution`` and ``.solution_info``
# attributes, respectively. Optimized risk scores are accessible from the ``.scores`` attribute.
# Reports may be generated using the ``.report()`` method.
#

rs.scores

###################################################################################################

rs.report()

# sphinx_gallery_start_ignore
from plotly.io import show
show(rs.report())
# sphinx_gallery_end_ignore