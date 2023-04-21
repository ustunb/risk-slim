"""Tests for lookup log loss functions."""

from itertools import product
import numpy as np


from riskslim.loss_functions.lookup_log_loss import (
    get_loss_value_table, get_prob_value_table, get_loss_value_and_prob_tables,
    log_loss_value, log_loss_value_from_scores, log_loss_value_and_slope,
)


def test_get_loss_value_table():
    """Test loss table."""

    min_score = np.arange(0, 5)
    offset = np.arange(1, 10)

    for _min, _off in product(min_score, offset):
        _max = _min + _off
        loss_value_table, lookup_offset = get_loss_value_table(_min, _max)

        assert lookup_offset == -_min
        assert len(loss_value_table) == (_max - _min) + 1
        # Monotonically decreasing
        assert np.all(np.diff(loss_value_table) < 0)


def test_get_prob_value_table():
    """Test probability table."""
    min_score = np.arange(0, 5)
    offset = np.arange(1, 10)

    for _min, _off in product(min_score, offset):
        _max = _min + _off
        prob_value_table, lookup_offset = get_prob_value_table(_min, _max)

        assert len(prob_value_table) == (_max - _min) + 1
        # Monotonically increasing
        assert np.all(np.diff(prob_value_table) > 0)


def get_loss_value_and_prob_tables():
    """Test loss and probability table."""
    min_score = np.arange(0, 5)
    offset = np.arange(1, 10)

    for _min, _off in product(min_score, offset):
        _max = _min + _off

        loss_value_table, prob_value_table, lookup_offset = \
            get_loss_value_and_prob_tables(_min, _max)

        _loss_value_table, _lookup_offset = get_loss_value_table(_min, _max)
        _prob_value_table, _ = get_prob_value_table(_min, _max)

        assert np.all(loss_value_table == _loss_value_table)
        assert np.all(prob_value_table == _prob_value_table)
        assert lookup_offset == _lookup_offset


def log_loss_value(generated_normal_data):
    """Test accuracy log loss lookup."""
    # Get simulated data from fixture
    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho']

    min_score = -100
    max_score = 100

    loss_value_table, lookup_offset = get_loss_value_table(min_score, max_score)

    loss = log_loss_value(Z, rho, loss_value_table, lookup_offset)

    assert loss >= loss_value_table.min()
    assert loss <= loss_value_table.max()

    # Free pointer to prevent malloc error
    del loss_value_table, lookup_offset


def log_loss_value_from_scores(generated_normal_data):
    """Test loss value from scores."""
    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho']

    min_score = -100
    max_score = 100

    loss_value_table, lookup_offset = get_loss_value_table(min_score, max_score)

    loss = log_loss_value_from_scores(Z.dot(rho), loss_value_table, lookup_offset)

    # Ensure implementation is the same between functions
    assert loss == log_loss_value(Z, rho, loss_value_table, lookup_offset)

    # Free pointer to prevent malloc error
    del loss_value_table, lookup_offset


def log_loss_value_and_slope(generated_normal_data):
    """Test loss and slope."""
    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho']

    min_score = -10000
    max_score = 10000

    loss_value_table, prob_value_table, lookup_offset = get_loss_value_and_prob_tables(
        min_score, max_score
    )

    loss, slope = log_loss_value_and_slope(
        Z, rho, loss_value_table, prob_value_table, lookup_offset
    )

    # Step
    step_size = .01
    rho_step = rho-step_size*slope

    loss_step = log_loss_value(
        Z, rho_step, loss_value_table, lookup_offset
    )

    # Loss should decrease after stepping
    assert loss_step < loss
