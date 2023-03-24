"""Tests for (fast) log loss functions."""

import pytest
import numpy as np

from riskslim.loss_functions import log_loss
from riskslim.loss_functions import fast_log_loss


@pytest.mark.parametrize(
    'log_loss_value', [log_loss.log_loss_value, fast_log_loss.fast_log_loss_value]
)
def test_log_loss_value(log_loss_value):
    """Test log loss accuracy for small weights."""
    for d in range(1, 101):

        Z = np.ones((d, d), dtype=np.float64, order='F')
        r = np.random.uniform(-1, 1)
        rho = np.ones(d, order='F') + r

        np.testing.assert_approx_equal(
            log_loss_value(Z, rho), np.log1p(np.exp(-(1+r) * d))
        )


@pytest.mark.parametrize(
    'log_loss_value_and_slope', [
        log_loss.log_loss_value_and_slope, fast_log_loss.log_loss_value_and_slope
    ]
)
def test_log_loss_value_and_slope(generated_class_data, log_loss_value_and_slope):
    """Test accuracy of the direction of log loss slope."""
    # Get simulated data from fixture
    Z = generated_class_data['Z']
    rho = generated_class_data['rho']

    # Loss
    loss, slope = log_loss_value_and_slope(Z, rho)

    # Step
    step_size = .1
    loss_step = log_loss_value(Z, rho-step_size*slope)

    assert loss_step < loss


@pytest.mark.parametrize(
    'log_loss_value_from_scores', [
        log_loss.log_loss_value_from_scores, fast_log_loss.log_loss_value_from_scores
    ]
)
def test_log_loss_value_from_scores(generated_class_data, log_loss_value_from_scores):
    """Test log loss from scores."""
    # Get simulated data from fixture
    Z = generated_class_data['Z']
    rho = generated_class_data['rho']

    # Ensure implementation is the same between functions
    assert log_loss_value_from_scores(Z.dot(rho)) == log_loss.log_loss_value(Z, rho)


def test_log_probs(generated_class_data):
    """Test log probabiltiies."""
    # Get simulated data from fixture
    Z = generated_class_data['Z']
    rho = generated_class_data['rho']
    y = generated_class_data['y']

    # Convert y \in {-1, 1} to y \in {0, 1}
    y_binary = y[:, 0].copy()
    y_binary[len(y_binary)//2:] = 0

    # Ensure probabilities align with labels
    assert np.all(log_probs(Z, rho) == y_binary)
