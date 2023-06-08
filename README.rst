=========
risk-slim
=========

risk-slim is a machine learning method to fit simple customized risk scores in python.

Background
----------

Risk scores let users make quick risk predictions by adding and subtracting a few small numbers (see e.g., 500 + medical risk scores at `mdcalc.com <https://www.mdcalc.com/>`_.

Here is a risk score for ICU risk prediction from our `paper <http://www.berkustun.com/docs/ustun_2017_optimized_risk_scores.pdf>`_.

.. image:: https://raw.githubusercontent.com/ustunb/risk-slim/master/docs/images/risk_score_seizure.png
  :width: 480
  :height: 360

Video
-----

.. raw:: html

	<iframe width="560" height="315" src="https://www.youtube.com/embed/WQDVejk17Aw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Reference
---------

If you use risk-slim in your research, we would appreciate a citation to the following paper: `bibtex <https://github.com/ustunb/risk-slim/blob/master/docs/references/ustun2019riskslim.bib>`_!

.. raw:: html

  <a href="http://jmlr.org/papers/v20/18-615.html" target="_blank">Learning Optimized Risk Scores</a> <br>
  Berk Ustun and Cynthia Rudin<br>
  Journal of Machine Learning Research, 2019.

Installation
------------

Run the following snippet in a Unix terminal to install risk-slim and complete a test run.

.. code-block:: shell

  git clone https://github.com/ustunb/risk-slim
  cd risk-slim
  # Install in editable mode
  pip install -e .
  # Batch run
  bash batch/job_template.sh


Quickstart
----------

.. code-block:: python

  from riskslim import RiskSLIMClassifier, load_data_from_csv
  from riskslim.utils import open_file

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


Requirements
------------

risk-slim requires Python 3.5+ and CPLEX 12.6+. For instructions on how to download and install, click `here <https://github.com/ustunb/risk-slim/blob/master/docs/cplex_instructions.md>`_.

Contributing
------------

I'm planning to pick up development again in Fall 2020. I can definitely use a hand! If you are interested in contributing, please reach out!

Here's the current development roadmap:

- `sci-kit learn interface <http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator>`_
- support for open source solver in `python-mip <https://github.com/coin-or/python-mip>`_
- basic reporting tools (roc curves, calibration plots, model reports)
- documentation
- pip
