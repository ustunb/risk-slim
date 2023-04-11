"""Test loss consistency between functions."""

import numpy as np

import riskslim.loss_functions.fast_log_loss as fast
import riskslim.loss_functions.log_loss as normal
import riskslim.loss_functions.log_loss_weighted as weighted
import riskslim.loss_functions.lookup_log_loss as lookup


def test_compare_log_loss_value_from_scores(generated_data):
    """Compare cython to numpy loss computation."""

    # Unpack variabels from fixture
    min_score = generated_data['min_score']
    max_score = generated_data['max_score']
    loss_value_tbl = generated_data['loss_value_tbl']
    loss_tbl_offset = generated_data['loss_tbl_offset']

    # Assert correctness of log_loss from scores function
    for s in range(int(min_score), int(max_score)+1):

        scores = np.array(s, dtype=np.float64, ndmin=1)
        normal_value = normal.log_loss_value_from_scores(scores)
        cython_value = fast.log_loss_value_from_scores(scores)

        table_value = loss_value_tbl[s+loss_tbl_offset]

        lookup_value = lookup.log_loss_value_from_scores(
            scores, loss_value_tbl, loss_tbl_offset
        )

        assert(np.isclose(normal_value, cython_value, rtol=1e-06))
        assert(np.isclose(table_value, cython_value, rtol=1e-06))
        assert(np.isclose(table_value, normal_value, rtol=1e-06))
        assert(np.equal(table_value, lookup_value))


def test_compare_log_loss_value(generated_data):

    # Unpack variabels from fixture
    Z = generated_data['Z']
    rho = generated_data['rho']

    # Python implementations need to be 'C' aligned instead of D aligned
    Z_py = np.require(Z, requirements=['C'])
    rho_py = np.require(rho, requirements=['C'])
    loss_value_tbl = generated_data['loss_value_tbl']
    loss_tbl_offset = generated_data['loss_tbl_offset']
    prob_value_tbl = generated_data['prob_value_tbl']

    # Check values and cuts
    normal_cut = normal.log_loss_value_and_slope(Z_py, rho_py)
    cython_cut = fast.log_loss_value_and_slope(Z, rho)
    lookup_cut = lookup.log_loss_value_and_slope(
        Z, rho, loss_value_tbl, prob_value_tbl, loss_tbl_offset
    )

    assert np.isclose(cython_cut[0], lookup_cut[0])
    assert np.isclose(cython_cut[0], normal_cut[0])
    assert np.isclose(lookup_cut[0], normal_cut[0])

    assert np.all(np.isclose(cython_cut[1], lookup_cut[1]))
    assert np.all(np.isclose(cython_cut[1], normal_cut[1]))
    assert np.all(np.isclose(lookup_cut[1], normal_cut[1]))


def test_loss_normal_vs_weighted(generated_data):
    """Compare weighted and normal values, cuts, and scores."""

    # Unpack variabels from fixture
    y = generated_data['y']
    Z = generated_data['Z']
    rho = generated_data['rho']
    weights = generated_data['weights']

    # Python implementations need to be 'C' aligned instead of D aligned
    Z_py = np.require(Z, requirements=['C'])
    rho_py = np.require(rho, requirements=['C'])
    scores_py = Z_py.dot(rho_py)

    # Normal
    normal_value = normal.log_loss_value(Z_py, rho_py)
    normal_cut = normal.log_loss_value_and_slope(Z_py, rho_py)

    # Weighted
    def get_weighted(weights, Z_py, rho_py, scores_py):
        weighted_value = weighted.log_loss_value(
            Z_py, weights, np.sum(weights), rho_py
        )

        weighted_cut = weighted.log_loss_value_and_slope(
            Z_py, weights, np.sum(weights), rho_py
        )

        weighted_scores = weighted.log_loss_value_from_scores(
            weights, np.sum(weights), scores_py
        )
        return weighted_value, weighted_cut, weighted_scores

    weights = np.ones(len(y))

    weighted_value, weighted_cut, weighted_scores = \
        get_weighted(weights, Z_py, rho_py, scores_py)

    # Value
    assert(np.isclose(normal_value, weighted_value))
    assert(np.isclose(normal_value, weighted_scores))

    # Cut
    assert(np.isclose(normal_cut[0], weighted_cut[0]))
    assert(all(np.isclose(normal_cut[1], weighted_cut[1])))

    # Re weight
    weights = np.random.rand(len(y))

    weighted_value, weighted_cut, weighted_scores = \
        get_weighted(weights, Z_py, rho_py, scores_py)

    assert(np.isclose(weighted_value, weighted_scores))
    assert(np.isclose(weighted_value, weighted_cut[0]))
