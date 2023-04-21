"""Test callback classes."""

import pytest
import numpy as np
from riskslim.solution_pool import FastSolutionPool
from riskslim.coefficient_set import CoefficientSet
from riskslim.heuristics import discrete_descent, sequential_rounding
from riskslim import RiskSLIM
from riskslim.fit.callbacks import LossCallback, PolishAndRoundCallback


@pytest.mark.parametrize('cut_queue', [None, FastSolutionPool(12)])
@pytest.mark.parametrize('polish_queue', [None, FastSolutionPool(12)])
def test_losscallback(generated_normal_data, cut_queue, polish_queue):

    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']

    variable_names = generated_normal_data['variable_names']

    coef_set = CoefficientSet(variable_names)
    rs = RiskSLIM(coef_set=coef_set, L0_min=0, L0_max=10)

    # Set data attributes
    rs.X = X
    rs.y = y
    rs.variable_names = variable_names
    rs.outcome_name = None
    rs.sample_weights = None
    rs.init_fit()
    rs.warmstart()

    # Initialize solution queues
    rs.cut_queue = cut_queue
    rs.polish_queue = polish_queue


    loss_cb = rs.mip.register_callback(LossCallback)

    loss_cb.initialize(
        indices=rs.mip_indices,
        stats=rs.stats,
        settings=rs.settings,
        compute_loss_cut=rs.compute_loss_cut,
        get_alpha=rs.get_alpha,
        get_L0_penalty_from_alpha=rs.get_L0_penalty_from_alpha,
        initial_cuts=rs.initial_cuts,
        cut_queue=rs.cut_queue,
        polish_queue=rs.polish_queue,
        verbose=rs.verbose,
    )

    assert loss_cb.cut_queue is not None
    assert loss_cb.polish_queue is not None
    assert loss_cb.compute_loss_cut == rs.compute_loss_cut
    assert loss_cb.get_alpha == rs.get_alpha
    assert loss_cb.get_L0_penalty_from_alpha == rs.get_L0_penalty_from_alpha


def test_polish_and_round_callback(generated_normal_data):

    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']

    variable_names = generated_normal_data['variable_names']

    coef_set = CoefficientSet(variable_names)
    rs = RiskSLIM(coef_set=coef_set, L0_min=0, L0_max=10)

    # Set data attributes
    rs.X = X
    rs.y = y
    rs.variable_names = variable_names
    rs.outcome_name = None
    rs.sample_weights = None
    rs.init_fit()
    rs.warmstart()

    # Initialize solution queues
    rs.cut_queue = FastSolutionPool(12)
    rs.polish_queue = FastSolutionPool(12)

    polisher = lambda rho: discrete_descent(
        rho,
        rs.Z,
        rs.C_0,
        rs.rho_max,
        rs.rho_min,
        rs.get_L0_penalty,
        rs.compute_loss_from_scores,
        True,
    )

    rounder = lambda rho, cutoff: sequential_rounding(
        rho,
        rs.Z,
        rs.C_0,
        rs.compute_loss_from_scores_real,
        rs.get_L0_penalty,
        cutoff
    )

    polish_cb = rs.mip.register_callback(PolishAndRoundCallback)

    polish_cb.initialize(
        indices=rs.mip_indices,
        control=rs.stats,
        settings=rs.settings,
        cut_queue=rs.cut_queue,
        polish_queue=rs.polish_queue,
        get_objval=rs.get_objval,
        get_L0_norm=rs.get_L0_norm,
        is_feasible=rs.is_feasible,
        polishing_handle=polisher,
        rounding_handle=rounder,
    )

    assert polish_cb.get_objval == rs.get_objval
    assert polish_cb.get_L0_norm == rs.get_L0_norm
    assert polish_cb.is_feasible == rs.is_feasible
    assert polish_cb.polishing_handle == polisher
    assert polish_cb.rounding_handle == rounder
