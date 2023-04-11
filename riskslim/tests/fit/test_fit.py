"""Test RiskSlim fitting."""

from warnings import warn
import pytest
import numpy as np
from scipy import sparse
from cplex import Cplex
from riskslim.coefficient_set import CoefficientSet
from riskslim.data import Bounds, Stats
from riskslim.fit import RiskSLIM


@pytest.mark.parametrize('init_coef', [True, False])
def test_RiskSLIM_init(init_coef):
    """Test RiskSLIM initialization."""
    variable_names = ['variable_' + str(i) for i in range(10)]

    coef_set = CoefficientSet(variable_names) if init_coef else None

    L0_min=0
    L0_max=10

    rs = RiskSLIM(coef_set=coef_set, L0_min=L0_min, L0_max=L0_max)

    assert rs.L0_min == L0_min
    assert rs.L0_max == L0_max
    assert rs.X is None
    assert rs.y is None

    type_check = lambda i, expected_type : isinstance(i, expected_type)

    if init_coef:
        expected_type = (float, int, np.ndarray)
        assert rs.variable_names == variable_names
    else:
        expected_type = type(None)
        assert type_check(rs.variable_names, expected_type)

    assert type_check(rs.rho_lb, expected_type)
    assert type_check(rs.rho_ub, expected_type)
    assert type_check(rs.c0_value, expected_type)
    assert type_check(rs.L0_reg_ind, expected_type)
    assert type_check(rs.C_0, expected_type)
    assert type_check(rs.C_0_nnz, expected_type)


def test_RiskSLIM_init_fit(generated_normal_data):
    """Test RiskSLIM fit initalization."""
    X = generated_normal_data['X'][0]
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']

    coef_set = CoefficientSet(variable_names)

    rs = RiskSLIM(coef_set=coef_set, L0_min=0, L0_max=10)

    # Load data into attribute
    #   this is normally done in .fit
    rs.X = X
    rs.y = y
    rs.variable_names = None
    rs.outcome_name = None
    rs.sample_weights = None

    rs.init_fit()

    # Checks
    assert isinstance(rs.coef_set, CoefficientSet)
    assert isinstance(rs.rho_lb, np.ndarray)
    assert isinstance(rs.rho_lb, np.ndarray)

    assert isinstance(rs._mip, Cplex)
    assert isinstance(rs._mip_indices, dict)

    assert isinstance(rs.bounds, Bounds)
    assert isinstance(rs.stats, Stats)

    assert (rs._Z.shape == rs.X.shape)


@pytest.mark.parametrize('update_settings', [True, False])
def test_RiskSLIM_warmstart(generated_normal_data, update_settings):
    """Test RiskSLIM fitting."""
    X = generated_normal_data['X']
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']

    coef_set = CoefficientSet(variable_names)

    n_iters = 25

    # Settings
    for seed in range(n_iters):

        rs = RiskSLIM(coef_set=coef_set, L0_min=0, L0_max=5)

        # Constraints
        ub = np.array([5.] * len(variable_names))
        lb = np.array([-5.] * len(variable_names))

        lb[0] = 0.
        ub[0] = 0.

        coef_set = CoefficientSet(variable_names=variable_names, lb=lb, ub=ub)

        # Load data into attribute
        #   this is normally done in .fit
        rs.X = X[seed]
        rs.y = y
        rs.variable_names = None
        rs.outcome_name = None
        rs.sample_weights = None

        # Initalize fitting procedure
        rs.init_fit()

        # Test warm start
        if update_settings:
            warmstart_settings = {'display_cplex_progress': False}
        else:
            warmstart_settings = None

        rs.warmstart(warmstart_settings)

        assert rs._has_warmstart
        assert len(rs.pool) > 0


def test_RiskSLIM_fit(generated_normal_data):
    """Test fitting RiskSLIM."""
    X = generated_normal_data['X']
    y = generated_normal_data['y']
    variable_names = generated_normal_data['variable_names']

    coef_set = CoefficientSet(variable_names)

    rs = RiskSLIM(coef_set=coef_set, L0_min=0, L0_max=5)

    # Size of problem
    n_columns = 12
    n_iters = 25
    c0_value = 1e-2

    # Settings
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        # LCPA Settings
        'max_runtime': 2,
        'max_tolerance': np.finfo('float').eps,
        'display_cplex_progress': False,
        'loss_computation': 'fast',
        # Initialization
        'initialization_flag': True,
        'init_max_runtime': 60.0,
        'init_max_coefficient_gap': 5,
        'init_display_progress': False,
        # CPLEX Solver Parameters
        'cplex_randomseed': 0,
        'cplex_mipemphasis': 0,
    }

    solutions = np.zeros((n_iters, n_columns), dtype=np.int8)

    for seed in range(n_iters):

        # Constraints
        ub = np.array([5.] * len(variable_names))
        lb = np.array([-5.] * len(variable_names))

        # Fix intercept at zero
        lb[0] = 0.
        ub[0] = 0.

        coef_set = CoefficientSet(variable_names=variable_names, lb=lb, ub=ub)

        # Initalize
        rs = RiskSLIM(coef_set=coef_set, L0_min=0, L0_max=10, settings=settings)

        # Fit
        rs.fit(X[seed], y)
        assert rs._fitted

        # Get solutions
        solutions[seed] = rs.solution_info['solution']

    # Test accuracy between computed and persistent solution
    solutions_persist = sparse.load_npz('riskslim/tests/solutions.npz').todense()
    percent_match = np.count_nonzero(solutions == solutions_persist)
    percent_match /= solutions.size
    percent_match *= 100

    if percent_match == 100.0:
        # Pass test
        pass
    elif percent_match >= 95:
        # Wiggle room for machine precision, cplex updates, etc
        warn(f'Persistence array match is {round(percent_match, 3)}%.')
    elif percent_match < 95:
        raise ValueError(f'Persistence array match is {round(percent_match, 3)}%.')
