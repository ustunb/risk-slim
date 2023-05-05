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

.. currentmodule:: riskslim.fit

.. autosummary::
   :toctree: generated/

   RiskSLIMOptimizer

   RiskSLIMClassifier

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
