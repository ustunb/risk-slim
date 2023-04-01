"""Test setup functions."""

import pytest
import numpy as np
import riskslim
from riskslim.setup_functions import (
    setup_loss_functions, setup_objective_functions, setup_penalty_parameters,
    get_loss_bounds
)


@pytest.mark.parametrize('loss_computation', ['fast', 'normal', 'weighted', 'lookup'])
def test_setup_loss_functions(generated_normal_data, loss_computation):
    """Test setting up loss functions."""
    data = generated_normal_data['data']
    rho = generated_normal_data['rho']

    coef_set = None

    if loss_computation == 'weighted':
        data['sample_weights'] = np.random.rand(len(data['X']))
    elif loss_computation == 'lookup':
        coef_set = riskslim.CoefficientSet(
            variable_names=data['variable_names'],
            lb=-10,
            ub=10,
            sign=0
        )

        coef_set.update_intercept_bounds(X=data['X'], y=data['Y'], max_offset=0)

    # Setup loss
    (Z, compute_loss, compute_loss_cut, compute_loss_from_scores, compute_loss_real,
        compute_loss_cut_real, compute_loss_from_scores_real) = setup_loss_functions(
            data, coef_set, loss_computation=loss_computation
        )

    # Tests
    assert np.all(Z == data['X'] * data['Y'])

    assert compute_loss(rho) == compute_loss_real(rho)
    assert compute_loss_from_scores(Z.dot(rho)) == compute_loss_from_scores_real(Z.dot(rho))

    loss, slope = compute_loss_cut(rho)
    loss_real, slope_real = compute_loss_cut_real(rho)

    assert loss == loss_real
    assert np.all(slope == slope_real)


def test_setup_objective_functions(generated_normal_data):
    """Test setting up objective functions."""
    data = generated_normal_data['data']
    Z = generated_normal_data['Z']
    rho = generated_normal_data['rho']
    rho_true = generated_normal_data['rho_true']

    coef_set = riskslim.CoefficientSet(
        variable_names=data['variable_names'],
        lb=-10,
        ub=10,
        sign=0
    )

    coef_set.update_intercept_bounds(X=data['X'], y=data['Y'], max_offset=0)

    c0_value, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(
        coef_set, c0_value=1e-6
    )

    compute_loss = lambda rho: riskslim.loss_functions.log_loss.log_loss_value(Z, rho)

    (get_objval, get_L0_norm, get_L0_penalty, get_alpha, get_L0_penalty_from_alpha) = \
        setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz)

    assert get_objval(rho) > get_objval(rho_true)
    assert get_L0_norm(rho) == len(data['X'][0]) - 1
    assert get_L0_penalty(rho) > 0
    assert np.all(get_alpha(rho) == rho[0])
    assert get_L0_penalty_from_alpha(get_alpha(rho)) == get_L0_penalty(rho)


@pytest.mark.parametrize('use_coef_set', [True, pytest.mark.xfail(False)])
def test_setup_penalty_parameters(generated_normal_data, use_coef_set):
    """Test setting up penalty parameters."""
    data = generated_normal_data['data']

    coef_set = None

    if use_coef_set:
        coef_set = riskslim.CoefficientSet(
            variable_names=data['variable_names'],
            lb=-10,
            ub=10,
            sign=0
        )

        coef_set.update_intercept_bounds(X=data['X'], y=data['Y'], max_offset=0)

    c0_value, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(
        coef_set, c0_value=1e-6
    )

    assert isinstance(c0_value, float)
    assert c0_value > 0
    assert len(C_0) == len(data['X'][0])

    assert len(L0_reg_ind) == len(data['X'][0])
    assert np.all(L0_reg_ind[1:])
    assert not L0_reg_ind[0]

    assert len(C_0_nnz) == len(data['X'][0]) - 1


def test_get_loss_bounds(generated_normal_data):
    """Test loss bounds."""
    Z = generated_normal_data['Z']
    rho_ub = 10
    rho_lb = .1
    L0_reg_ind = np.ones(len(Z[0]), dtype=bool)
    L0_reg_ind[0] = False

    min_loss, max_loss = get_loss_bounds(
        Z, rho_ub, rho_lb, L0_reg_ind, L0_max=float('nan')
    )

    assert min_loss == 0
    assert max_loss > 0 and max_loss < np.inf
