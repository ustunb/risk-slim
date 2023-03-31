"""Configuration file for pytest."""

import os
import pytest

import numpy as np

from riskslim.coefficient_set import get_score_bounds
from riskslim.setup_functions import _setup_training_weights
import riskslim.loss_functions.lookup_log_loss as lookup


@pytest.fixture(scope='module')
def generated_data():

    np.random.seed(seed=0)
    n_rows = 1000000
    n_cols = 20
    rho_ub = 100
    rho_lb = -100

    def generate_binary_data(n_rows=1000000, n_cols=20):
        X = np.random.randint(low=0, high=2, size=(n_rows, n_cols))
        y = np.random.randint(low=0, high=2, size=(n_rows, 1))
        pos_ind = y == 1
        y[~pos_ind] = -1
        return X, y

    def generate_integer_model(n_cols=20, rho_ub=100, rho_lb=-100, sparse_pct=0.5):
        rho = np.random.randint(low=rho_lb, high=rho_ub, size=n_cols)
        rho = np.require(rho, dtype=np.float64, requirements=['F'])
        nnz_count = int(sparse_pct * np.floor(n_cols / 2))
        set_to_zero = np.random.choice(range(0, n_cols), size=nnz_count, replace=False)
        rho[set_to_zero] = 0.0
        return rho

    # Initialize data matrix X and label vector y
    X, y = generate_binary_data(n_rows, n_cols)

    Z = X * y
    Z = np.require(Z, requirements=['F'], dtype=np.float64)

    rho = generate_integer_model(n_cols, rho_ub, rho_lb)

    L0_reg_ind = np.ones(n_cols, dtype='bool')
    L0_reg_ind[0] = False

    Z_min = np.min(Z, axis=0)
    Z_max = np.max(Z, axis=0)

    # Setup weights
    weights = _setup_training_weights(y, w_pos=1.0, w_neg=1.0, w_total_target=2.0)

    # Create lookup table
    min_score, max_score = get_score_bounds(Z_min, Z_max, rho_lb, rho_ub, L0_max=n_cols)

    loss_value_tbl, prob_value_tbl, loss_tbl_offset = \
            lookup.get_loss_value_and_prob_tables(min_score, max_score)

    loss_tbl_offset = int(loss_tbl_offset)

    generated = {
        'X': X,
        'y': y,
        'Z': Z,
        'rho': rho,
        'min_score': min_score,
        'max_score': max_score,
        'weights': weights,
        'loss_value_tbl': loss_value_tbl,
        'loss_tbl_offset': loss_tbl_offset,
        'prob_value_tbl': prob_value_tbl
    }

    yield generated


@pytest.fixture(scope='module')
def generated_class_data():
    """Generate data with a solution."""
    n_per_class = 10000
    dims = 100

    # Simulate features
    np.random.seed(0)
    X = np.random.randint(1, int(dims), size=(n_per_class*2, dims))
    X = np.sort(X, axis=1)
    X[n_per_class:] = X[n_per_class:][:, ::-1]

    # Assign labels
    y = np.ones((2*n_per_class, 1))
    y[n_per_class:] = -1

    # Training data and weights
    Z = X * y
    Z = np.asfortranarray(Z)

    rho = np.ones(dims, order='F')

    yield {'X':X, 'y':y, 'Z':Z, 'rho':rho}
