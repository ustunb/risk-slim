"""Test utility functions."""

import os
import tempfile
from itertools import cycle
import logging
import pytest
import numpy as np
from riskslim.utils import (
    load_data_from_csv, check_data, print_model, setup_logging, print_log,
    validate_settings, is_integer, cast_to_integer
)


@pytest.mark.parametrize('sample_weights_csv_file',
    [
        None,
        pytest.param('/bad/path', marks=pytest.mark.xfail(raises=IOError)),
    ]
)
def test_load_data_from_csv(sample_weights_csv_file):

    # Data
    data_name = "breastcancer"  # name of the data
    data_dir = os.getcwd() + '/examples/data/'  # directory where datasets are stored
    data_csv_file = data_dir + data_name + '_data.csv'  # csv file for the dataset

    folds = cycle(list(range(1, 6)))
    folds = [next(folds) for _ in range(683)]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for i in folds:
            f.write(str(i) + '\n')

    data = load_data_from_csv(
        dataset_csv_file=data_csv_file, sample_weights_csv_file=sample_weights_csv_file,
        fold_csv_file=f.name, fold_num=1
    )

    os.remove(f.name)

    assert isinstance(data['X'], np.ndarray)
    assert isinstance(data['y'], np.ndarray)
    assert len(data['X']) == len(data['y']) == len(data['sample_weights'])
    assert len(data['X'][0]) == len(data['variable_names'])
    assert isinstance(data['outcome_name'], str)


@pytest.mark.parametrize('y',
    [
        np.random.choice([-1, 1], 100).astype(int),
        np.ones(100).astype(int),
        np.ones(100).astype(int) * -1
    ]
)
@pytest.mark.parametrize('variable_names',
    [
        ['(Intercept)', *['var_' + str(i) for i in range(9)]],
        ['var_' + str(i) for i in range(10)]
    ]
)
@pytest.mark.parametrize('outcome_name',
    [
        'Outcome',
        pytest.param(0, marks=pytest.mark.xfail(raises=AssertionError))
    ]
)
@pytest.mark.parametrize('sample_weights',
    [
        np.ones(100),
        np.random.uniform(.1, .9, 100),
        pytest.param(0, marks=pytest.mark.xfail(raises=AssertionError))
    ]
)
def test_check_data(y, variable_names, outcome_name, sample_weights):

    X = np.random.rand(100, 10)
    X[:, 0] = 1.
    check = check_data(X, y, variable_names, outcome_name, sample_weights)

    assert check


@pytest.mark.parametrize('variable_names',
    [
        ['(Intercept)', *['var_' + str(i) for i in range(9)]],
        ['var_' + str(i) for i in range(10)]
    ]
)
@pytest.mark.parametrize('outcome_name', ['Outcome',None])
def test_print_model(variable_names, outcome_name):

    X = np.random.rand(100, 10)

    rho = np.random.rand(10)

    print_model(rho, X, variable_names, outcome_name)


@pytest.mark.parametrize('log_to_console', [True, False])
@pytest.mark.parametrize('log_file', [True, False])
def test_setup_logging(log_to_console, log_file):

    temp = None
    if log_file is True:
        temp = tempfile.NamedTemporaryFile(suffix='.log', mode='w')
        log_file = temp.name
    else:
        log_file = None

    logger = logging.getLogger()
    logger = setup_logging(logger, log_to_console=log_to_console, log_file=log_file)

    if temp is not None:
        temp.close()

    assert True


def test_print_log():

    print_log('msg', True)
    print_log('msg', False)


@pytest.mark.parametrize('settings', [None, {'b':2}])
def test_validate_settings(settings):

    defaults = {'a': 0, 'b': 1}
    validated_settings = validate_settings(settings, defaults)

    if settings is None:
        assert validated_settings['a'] == defaults['a']
        assert validated_settings['b'] == defaults['b']
    else:
        assert validated_settings['a'] == defaults['a']
        assert validated_settings['b'] == settings['b']


def test_is_integer():
    assert is_integer(np.array([0]))
    assert not is_integer(np.array([0.1]))


def test_cast_to_integer():

    int_arr = cast_to_integer(np.array([0.01, 1.01]))
    assert int_arr.dtype == np.float64
    assert int_arr[0] == 0.
    assert int_arr[1] == 1.
