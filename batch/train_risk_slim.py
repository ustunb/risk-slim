#!/usr/bin/python

"""
This file is to train a RiskSLIM classifier in a batch computing environment
It parses command line arguments

It can be called as

python train_risk_slim.py "${data_file}" "${fold_file}" "${fold_num}" "${save_file}" "${settings_file}"

where

data_file       csv file containing the training data
fold_file       csv file containing training folds (optional; if not included, then all data is used)
fold_num        integer between 0 to K to specify the # of the fold (optional; if not specified, then 0 so that all data is used)
results_file    file name for the save file; needs to be unique and not already exist on disk
settings_file   text file with name-value pairs for LCPA settings

Copyright (C) 2017 Berk Ustun
"""

import os
import sys
from os import path
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) )
import time
import argparse
import logging
import pickle
import numpy as np

from riskslim.helper_functions import load_data_from_csv, setup_logging, print_model
from riskslim.CoefficientSet import CoefficientSet
from riskslim.lattice_cpa import get_conservative_offset, run_lattice_cpa, DEFAULT_LCPA_SETTINGS

DEFAULT_SETTINGS = {
    'max_coefficient': 5,
    'max_L0_value': 5,
    'max_offset': 50,
}

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='train_risk_slim',
        description='Train a RiskSLIM classifier from the command shell',
        epilog='Copyright (C) 2017 Berk Ustun',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data',
                        type=str,
                        required=True,
                        help='csv file with training data')

    parser.add_argument('--results',
                        type=str,
                        required=True,
                        help='name of results file (must not already exist)')

    parser.add_argument('--cvindices',
                        type=str,
                        default='None',
                        help='csv file with indices for K-fold CV')

    parser.add_argument('--fold',
                        type=int,
                        default=0,
                        help='fold number')

    parser.add_argument('--timelimit',
                        type=int,
                        default=3600,
                        help='time limit on training in seconds')

    parser.add_argument('--log',
                        type=str,
                        default='None',
                        required=False,
                        help='name of the log file')

    parser.add_argument('--silent',
                        action='store_true',
                        required=False,
                        help='add to suppress logging to stderr')

    parser.add_argument('--settings',
                        type=str,
                        default='None',
                        help='text file containing additional settings for LCPA')

    args = parser.parse_args()
    parsed = vars(args)

    # setup logging
    log_to_console = not parsed['silent']
    log_file = parsed['log'] if parsed['log'] is not 'None' else None
    logger = logging.getLogger()
    logger = setup_logging(logger, log_to_console = True, log_file = log_file)
    logger.setLevel(logging.INFO)
    logger.info("running 'train_risk_slim.py'")
    logger.info("working directory: %r" % os.getcwd())
    logger.info("data_file: %r" % parsed['data'])
    logger.info("fold_file: %r" % parsed['cvindices'])
    logger.info("fold_num: %r" % parsed['fold'])
    logger.info("time_limit: %r" % parsed['timelimit'])
    logger.info("settings_file: %r" % parsed['results'])
    logger.info("results_file: %r" % parsed['settings'])

    #parse arguments
    data_file = parsed['data']
    fold_file = parsed['cvindices']
    fold_num = parsed['fold']
    max_runtime = parsed['timelimit']
    results_file = parsed['results']
    settings_file = parsed['settings']

    # check data_file exists
    if not os.path.isfile(data_file):
        logger.error("data_file %r not found on disk" % data_file)
        sys.exit(1)

    # check results_file does not exist
    if os.path.isfile(results_file):
        logger.error("results_file %r already exists)" % results_file)
        logger.error("either delete %r or choose a different name" % results_file)
        sys.exit(1)

    # check fold_file exists
    if not os.path.isfile(fold_file):
        fold_file = None
        fold_num = 0

    # check settings_file exists / or use default settings
    if os.path.isfile(settings_file):
        with open(settings_file) as f:
            settings = {l.strip() for l in f.readlines() if l}
    else:
        logger.warn("settings_file %r not found on disk" % settings_file)
        settings = {}

    # overwrite time limit on settings
    settings['timelimit'] = parsed['timelimit']


    # check if sample weights file was specified, if not set as None
    logger.info("loading data and sample weights")

    sample_weights_file = None
    if 'sample_weights_file' in settings:
        if os.path.isfile(settings['sample_weights_file']):
            sample_weights_file = settings['sample_weights_file']

    data = load_data_from_csv(dataset_csv_file=data_file,
                              sample_weights_csv_file=sample_weights_file,
                              fold_csv_file=fold_file,
                              fold_num=fold_num)
    N, P = data['X'].shape

    # initialize coefficient set and offset parameter
    logger.info("creating coefficient set and constraints")
    max_coefficient = settings['max_coefficient'] if 'max_coefficient' in settings else DEFAULT_SETTINGS['max_coefficient']
    max_L0_value = settings['max_L0_value'] if 'max_L0_value' in settings else DEFAULT_SETTINGS['max_L0_value']
    max_offset = settings['max_offset'] if 'max_offset' in settings else DEFAULT_SETTINGS['max_offset']

    coef_set = CoefficientSet(variable_names=data['variable_names'], lb=-max_coefficient, ub=max_coefficient, sign=0)
    conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
    max_offset = min(max_offset, conservative_offset)
    coef_set.set_field('lb', '(Intercept)', -max_offset)
    coef_set.set_field('ub', '(Intercept)', max_offset)
    coef_set.view()

    trivial_L0_max = P - np.sum(coef_set.C_0j == 0)
    max_L0_value = min(max_L0_value, trivial_L0_max)

    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set': coef_set,
    }

    # initialize lcpa_settings
    lcpa_settings = {key: settings[key] if key in settings else DEFAULT_LCPA_SETTINGS[key] for key in
                     DEFAULT_LCPA_SETTINGS}

    # fit RiskSLIM model using Lattice Cutting Plane Algorithm
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, lcpa_settings)

    # save output to disk
    results = {
        "date": time.strftime("%d/%m/%y", time.localtime()),
        "data_file": data_file,
        "fold_file": fold_file,
        "fold_num": settings_file,
        "results_file": results_file,
    }
    results.update(model_info)

    logger.info("saving results as pickle file")
    with open(results_file, 'wb') as outfile:
        pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("saved results to %r" % results_file)

    logger.info("finished training model")
    logger.info("quitting")
    sys.exit(0)