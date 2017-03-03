#!/usr/bin/env bash

#!/bin/bash

#This is a script to show how to train RiskSLIM
home_dir=$(pwd)                    #directory where code is stored
repo_dir=$(pwd)                    #directory where code is stored
data_dir="${repo_dir}/datasets"    #directory where data, sample weights and cvindices are stored
run_dir="${repo_dir}/scratch/run"  #directory where results will be saved as a pickle file

#required parameters
data_file="${data_dir}/breastcancer_data.csv"
cvindices_file="${data_dir}/breastcancer_folds.csv"
fold=0

#results file must have a unique name
results_file="${run_dir}/breastcancer_results.p"
rm "${results_file}"

#write settings to settings_file
settings_file="${run_dir}"

#now=$(date +"%m_%d_%Y_%H_%M_%S")
#run_log="${run_name}""_""${now}"".log"

python "${repo_dir}/batch/train_risk_slim.py"  \
    --data "${data_file}" \
    --results "${results_file}" \
    --cvindices "${cvindices_file}" \
    --fold "${fold}" \
    --settings "${settings_file}"

exit