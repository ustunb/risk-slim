"""Configuration file for pytest."""

import pickle

import numpy as np
import pytest

from generate_training_test_cases import REGENERATE_COMMAND, TEST_CASE_PATH, generate_data
from utils import generate_random_normal


def pytest_addoption(parser):
    """Register --report-dir for the opt-in browser test's screenshots and HTML."""
    parser.addoption(
        "--report-dir",
        default=None,
        help="Directory for report screenshots and HTML from `pytest -m browser`. Write it as "
        "--report-dir=DIR: with a space, pytest reads an existing DIR as a test path.",
    )


@pytest.fixture(scope="session")
def training_test_cases():
    """Load the cached training answers and regenerate each synthetic dataset from its seed."""
    if not TEST_CASE_PATH.exists():
        pytest.fail(
            f"Missing {TEST_CASE_PATH}. Restore it with `git restore tests/training_test_cases.pkl`.",
            pytrace=False,
        )
    with TEST_CASE_PATH.open("rb") as file_handle:
        store = pickle.load(file_handle)
    for case_id, record in store["synthetic_oracles"].items():
        data = generate_data(case_id)
        if data["data_digest"] != record["data_digest"]:
            pytest.fail(
                f"{case_id} no longer regenerates the data its cached answers were computed on. "
                f"Rebuild it with `{REGENERATE_COMMAND} --case {case_id}`.",
                pytrace=False,
            )
        record["X"] = data["X"]
        record["y"] = data["y"]
    return store


@pytest.fixture(scope="module")
def generated_normal_data():
    """Generate data with a solution."""

    # Size of problem
    n_columns = 12
    n_rows = 200
    n_targets = 4
    n_iters = 25

    random_state = np.random.get_state()
    try:
        X = np.zeros((n_iters, n_rows, n_columns))
        weights_true = np.zeros((n_iters, n_columns))
        for i, seed in enumerate(range(n_iters)):
            # Simulate data
            _data, _weights_true = generate_random_normal(n_rows, n_columns, n_targets, seed)

            # Track features and true weights
            X[i] = _data["X"]
            weights_true[i] = _weights_true
    finally:
        np.random.set_state(random_state)

    # Labels
    y = np.ravel(_data["y"])

    weights = np.ones(n_columns)

    Z = X * y[:, None]

    names = ["var_" + str(i).zfill(2) for i in range(n_columns)]

    yield {
        "X": X,
        "y": y,
        "Z": Z,
        "weights": weights,
        "weights_true": weights_true,
        "variable_names": names,
        "outcome_name": _data["outcome_name"],
    }
