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
import pickle
import numpy as np
from riskslim.helper_functions import load_data_from_csv, print_model, print_log
from riskslim.CoefficientSet import CoefficientSet
from riskslim.lattice_cpa import get_conservative_offset, run_lattice_cpa, DEFAULT_LCPA_SETTINGS

DEFAULT_BATCH_SETTINGS = {
    'max_coefficient': 5,
    'max_L0_value': 5,
    'max_offset': 50,
}

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='train_risk_slim',
        description='This file is to train a RiskSLIM classifier in a batch computing environment',
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

    parser.add_argument('--settings',
                        type=str,
                        default='None',
                        help='text file containing additional settings for LCPA')

    args = parser.parse_args()
    parsed = vars(args)
    data_file = parsed['data']
    results_file = parsed['results']
    fold_file = parsed['cvindices']
    fold_num = parsed['fold']
    settings_file = parsed['settings']

    # print stuff to screen
    print_log("running 'train_risk_slim.py'")
    print_log("working directory: %r" % os.getcwd())
    print_log("data_file: %r" % data_file)
    print_log("fold_file: %r" % fold_file)
    print_log("fold_num: %r" % fold_num)
    print_log("settings_file: %r" % settings_file)
    print_log("results_file: %r" % settings_file)

    # check data_file exists
    if not os.path.isfile(data_file):
        print_log("data_file %r not found on disk" % data_file)
        sys.exit(1)

    # check results_file does not exist
    if os.path.isfile(results_file):
        print_log("ERROR: results_file %r already exists\nDelete the old file, or choose a different name" % results_file)
        sys.exit(1)

    # check fold_file exists
    if not os.path.isfile(fold_file):
        fold_file = None
        fold_num = 0

    # check settings_file exists / or use default settings
    if os.path.isfile(settings_file):
        with open(settings_file) as f:
            run_settings = {l.strip() for l in f.readlines() if l}
    else:
        print_log('settings_file %r not found on disk' % settings_file)
        run_settings = {}

    # check if sample weights file was specified, if not set as None
    sample_weights_file = None
    if 'sample_weights_file' in run_settings:
        if os.path.isfile(run_settings['sample_weights_file']):
            sample_weights_file = run_settings['sample_weights_file']

    data = load_data_from_csv(dataset_csv_file=data_file,
                              sample_weights_csv_file=sample_weights_file,
                              fold_csv_file=fold_file,
                              fold_num=fold_num)
    N, P = data['X'].shape

    # model form constraints
    max_coefficient = run_settings['max_coefficient'] if 'max_coefficient' in run_settings else DEFAULT_BATCH_SETTINGS[
        'max_coefficient']
    max_L0_value = run_settings['max_L0_value'] if 'max_L0_value' in run_settings else DEFAULT_BATCH_SETTINGS[
        'max_L0_value']
    max_offset = run_settings['max_offset'] if 'max_offset' in run_settings else DEFAULT_BATCH_SETTINGS['max_offset']

    # initialize coefficient set and offset parameter
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
    lcpa_settings = {key: run_settings[key] if key in run_settings else DEFAULT_LCPA_SETTINGS[key] for key in
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

    with open(results_file, 'wb') as outfile:
        pickle.dump(results, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    with open(results_file, "rb") as infile:
        loaded_results = pickle.load(infile)

    print_log("finished training model")
    print_log("quitting")
    sys.exit(0)