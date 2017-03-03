#!/bin/bash

# This is a template to show how to train RiskSLIM from the Bash shell
# Change it to run RiskSLIM on a batch computing environment (AWS Batch)
#
# To test the script, run it from the risk-slim directory using the command:
# `bash batch/job_template.sh`
#
#
# Recommended Directory Structure
#
#   risk-slim/          
#       └──batch/
#           └──data/           location of CSV files for data (ignored in .gitignore)
#           └──logs/           directory where log files are printed out (ignored in .gitignore)
#           └──results/        directory where results files are stored (ignored in .gitignore)
#       └──doc/
#       └──examples/           
#       └──riskslim/           directory where code is stored (do not change this to be able to pull from GitHub)
#       └──setup.py

#directories
repo_dir=$(pwd)"/"
data_dir="${repo_dir}examples/data/"    #change to /batch/data/ for your own data
results_dir="${repo_dir}batch/results/"
log_dir="${repo_dir}batch/logs/"

#job parameters
data_name="breastcancer"
data_file="${data_dir}/${data_name}_data.csv"
cvindices_file="${data_dir}/${data_name}_cvindices.csv"
fold=0
timelimit=60

#results_file and log_file must have a UNIQUE name for each job
#otherwise we may overwrite existing results *or* run into trouble when running in parallel
#for safety, train_risk_slim.py does not run a results_file already exists on disk
#this does not handle issues when jobs running in parallel overwrite each otehr
run_name="${data_name}_fold_${fold}"

results_file="${results_dir}${run_name}.results"
rm -f "${results_file}"

now=$(date +"%m_%d_%Y_%H_%M_%S")
log_file="${log_dir}${run_name}_${now}.log"

#create directories that do not exist
mkdir -p "${results_dir}"
mkdir -p "${log_dir}"

#write settings to settings_file
settings_file="${results_dir}${run_name}.settings"

python "${repo_dir}/batch/train_risk_slim.py"  \
    --data "${data_file}" \
    --results "${results_file}" \
    --cvindices "${cvindices_file}" \
    --fold "${fold}" \
    --timelimit "${timelimit}" \
    --settings "${settings_file}" \
    --log "${log_file}"

exit