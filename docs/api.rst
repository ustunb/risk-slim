.. _api:

===
API
===

API reference for the riskslim module.

Table of Contents
=================

.. contents::
   :local:
   :depth: 1

Model
-----

Object for optimizing risk scores.

.. currentmodule:: riskslim.optimizer

.. autosummary::
   :toctree: generated/

   RiskSLIMOptimizer

Scikit-learn compatible interface for the optimizer.

.. currentmodule:: riskslim.classifier

.. autosummary::
   :toctree: generated/

   RiskSLIMClassifier

Scores
------

Risk scores, derived metrics, and reporting.

.. currentmodule:: riskslim.risk_scores

.. autosummary::
   :toctree: generated/

   RiskScores


Coefficients
------------

Class to represent and specify constraints on coefficients of input variables.

.. currentmodule:: riskslim.coefficient_set

.. autosummary::
   :toctree: generated/

   CoefficientSet




MIP
---

RiskSLIM MIP formulation with CPLEX.

.. currentmodule:: riskslim.mip

.. autosummary::
   :toctree: generated/

   create_risk_slim
   set_cplex_mip_parameters
   set_cpx_display_options
   add_mip_starts
   cast_mip_start
   convert_to_risk_slim_cplex_solution

Loss Functions
--------------

.. currentmodule:: riskslim.loss_functions

.. autosummary::
   :toctree: generated/

   log_loss.log_loss_value
   log_loss.log_loss_value_and_slope
   log_loss.log_loss_value_from_scores
   log_loss.log_probs


Callbacks
---------

Calls when CPLEX finds an integer feasible solution.

.. currentmodule:: riskslim.callbacks

.. autosummary::
   :toctree: generated/

   LossCallback
   PolishAndRoundCallback

