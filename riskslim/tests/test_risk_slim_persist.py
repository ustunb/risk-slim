"""Test the persistence of solutions across development."""

from warnings import warn
import numpy as np
from scipy import sparse
import riskslim

from .utils import generate_random_normal


def test_risk_slim_persistence():
    """Test the persistency of solutions from commit to commit, or PR to PR, to prevent
    breaking changes to the algorithm.

    Notes
    -----
    This test runs lattice cpa for 25 simulations of random normals centered at either
    -i or +i depending on class, and then compares the computed solution to that saved
    in solutions.npz.
    """
    # Size of problem
    n_columns = 12
    n_rows = 200
    n_targets = 4
    n_iters = 25

    # Constraints
    max_coefficient = 10
    max_L0_value = 10
    max_offset = 0
    c0_value = 1e-2

    varnames = ['var_' + str(i).zfill(2) for i in range(n_columns-1)]
    varnames.insert(0, '(Intercept)')

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

    y = np.ones((n_rows, 1), dtype=np.int32)
    y[n_rows//2:, 0] = -1

    for seed in range(n_iters):

        # Simulate data
        data, _ = generate_random_normal(n_rows, n_columns, n_targets, seed)

        # Contraints
        coef_set = riskslim.CoefficientSet(
            variable_names=varnames,
            lb=-max_coefficient,
            ub=max_coefficient,
            sign=0
        )
        coef_set.update_intercept_bounds(data['X'], y=data['Y'], max_offset=max_offset)
        constraints = {'L0_min': 0, 'L0_max': max_L0_value, 'coef_set': coef_set}

        # Fit
        model_info, _, _ = riskslim.run_lattice_cpa(data, constraints, settings)

        solutions[seed] = model_info['solution']

    # Load persistent solutions
    solutions_persist = sparse.load_npz('riskslim/tests/solutions.npz')

    # Test accuracy between computed and persistent solution
    percent_match = np.count_nonzero(solutions == solutions_persist.todense())
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
