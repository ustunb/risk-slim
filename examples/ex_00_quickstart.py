"""
00. Quickstart
"""

from riskslim import RiskSLIMClassifier, load_data_from_csv
from riskslim.utils import open_file
from pathlib import Path

# Load Data
url = "https://raw.githubusercontent.com/ustunb/risk-slim/master/examples/data/"
url += "breastcancer_data.csv"

data = load_data_from_csv(url)

# Initialize Model
rs = RiskSLIMClassifier(
    max_size = 5, # max model size (number of non-zero coefficients)
    max_coef = 5, # value of largest/smallest coefficient
    variable_names = data["variable_names"],
    outcome_name = "poisonous",
    verbose = False
)

# Fit
rs.fit(data["X"], data["y"])

# Create Report
report_file = rs.create_report(file_name = 'report.html')
open_file(report_file)