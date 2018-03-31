#!/usr/bin/python
if __name__ == '__main__':

    import os
    import glob
    import subprocess
    #from debugging import ipsh

    #testing parameters
    script_name = "rounding_methods_script.py"
    #script_name = "basic_risk_slim_script.py"
    #script_name = "custom_risk_slim_script.py"
    create_new_execution_file = True
    execute_script = True
    clean_up_settings_file = False
    clean_up_results_file = False

    #test case parameters
    data_name = "ptsd2"
    #data_name = "breastcancer_N1000e3_d30"
    xtra_id = "QS4"
    hcon_id = "U002"
    fold_id = "K10N01"
    inner_fold_id = "NONE"
    fold_num = 0
    sample_weight_id = "NONE"
    w_pos = 1.0
    w_neg = 1.0
    display_cplex_progress = True

    results_file_suffix = "results_raw"
    save_results_as_rdata = True
    n_cores = 1

    cplex_parameters = {
        'randomseed': 0,
        'repairtries': 20,
        'mipemphasis': 0,
        'nodefilesize': (120 * 1024) / n_cores,
        'poolsize': 100,
        'poolreplace': 2,
    }

    settings = {
        # rounded
        #'lars_elasticnet_file': data_name + "_F_" + fold_id + "_RSPP_processed.RData",
        'add_custom_constraints': "TRUE",
        "load_previous_results": False,
        'only_use_best_RSLP_solution': 'TRUE',
        'use_custom_coefficient_set': False,
        'max_coefficient': 5,
        'max_offset': -1,
        'max_L0_value': -1,
        'c0_value': 0.00001,
        #
        'loss_computation': 'normal',
        'max_runtime': 30.0,
        'max_tolerance': 0.000001,
        'max_iterations': 100000,
        #
        'store_stats': False,
        'display_cplex_progress': True,
        'display_updates': False,
        'display_updates_at_rounding': False,
        'display_updates_at_polishing': False,
        #
        'tight_formulation': False,
        'polish_flag': False,
        'round_flag': False,
        'polish_rounded_solutions': False,
        'add_cuts_at_heuristic_solutions': False,
        'global_bound_flag': False,
        'local_bound_flag': False,
        #
        'polishing_ub_to_objval_relgap': 0.1,
        'polishing_max_runtime': 10.0,
        'polishing_max_solutions': 5.0,
        'polishing_min_cuts': 0,
        'polishing_max_cuts': float('inf'),
        'polishing_min_relgap': 5.0,
        'polishing_max_relgap': float('inf'),
        #
        'rounding_min_cuts': 0,
        'rounding_max_cuts': 20000,
        'rounding_min_relgap': 0.2,
        'rounding_max_relgap': float('inf'),
        'rounding_ub_to_objval_relgap': float('inf'),
    }

    warmstart_settings = {
        'use_cvx': True,
        'cvx_display_progress': False,
        'cvx_display_cplex_progress': False,
        'cvx_save_progress': True,
        #
        'cvx_max_runtime': 10.0,
        'cvx_max_runtime_per_iteration': 30.0,
        'cvx_max_cplex_time_per_iteration': 10.0,
        'cvx_max_iterations': 10000,
        'cvx_max_tolerance': 0.0001,
        #
        'cvx_use_sequential_rounding': True,
        'cvx_sequential_rounding_max_runtime': 30.0,
        'cvx_sequential_rounding_max_solutions': 5,
        #
        'use_ntree': False,
        'ntree_display_progress': True,
        'ntree_display_cplex_progress': False,
        'ntree_save_progress': True,
        #
        'ntree_max_runtime_per_iteration': 2.0,
        'ntree_max_cplex_time_per_iteration': 2.0,
        'ntree_max_runtime': settings['max_runtime'],
        'ntree_max_iterations': 1000000,
        'ntree_max_tolerance': settings['max_tolerance'],
        #
        'polishing_after': False,
        'polishing_max_runtime': 30.0,
        'polishing_max_solutions': float('inf'),
    }

    #directories
    comp_name = "berkmac"
    slim_dir = "/Users/berk/Dropbox (MIT)/Research/SLIM/"
    data_dir = slim_dir + "Data/"
    python_dir = slim_dir + "Python/"

    print "Testing %s on %s" % (script_name, comp_name)
    print "Current directory: %s" % (os.getcwd())

    if settings['use_custom_coefficient_set']:
        hcon_id = "U000"

    #create files
    if create_new_execution_file:
        header_file_name = slim_dir + 'header_file.sh'
        body_file_name = slim_dir + 'onetree_test_script_body.sh'
        test_file_name = 'test_script.sh'
        full_test_file_name = slim_dir + test_file_name

        #create header file
        with open(header_file_name, 'w') as header_file:
            header_file.write("#!/bin/bash\n")
            header_file.write("comp_name='%s'\n" % comp_name)
            header_file.write("data_name='%s'\n" % data_name)
            header_file.write("fold_id='%s'\n" % fold_id)
            header_file.write("fold_num='%s'\n" % fold_num)
            header_file.write("inner_fold_id='%s'\n" % inner_fold_id)
            header_file.write("w_pos=%1.3f\n" % w_pos)
            header_file.write("sample_weight_id='%s'\n" % sample_weight_id)
            header_file.write("hcon_id='%s'\n" % hcon_id)
            header_file.write("xtra_id='%s'\n" % xtra_id)
            header_file.write("results_file_suffix='%s'\n" % results_file_suffix)
            header_file.write("script_name='%s'\n" % script_name)
            header_file.write("run_script_name='%s'\n" % script_name)

            for key in settings.keys():
                print 'writing %s' % key
                header_file.write("RiskSLIM_%s='%s'\n" % (key, settings[key]))

            for key in cplex_parameters.keys():
                print 'writing %s' % key
                header_file.write("RiskSLIM_cplex_%s='%s'\n" % (key, cplex_parameters[key]))

            for key in warmstart_settings.keys():
                print 'writing %s' % key
                header_file.write("RiskSLIM_%s='%s'\n" % (key, warmstart_settings[key]))

        #add body to header
        with open(full_test_file_name, 'w') as outfile:
            for fname in [header_file_name, body_file_name]:
                with open(fname) as infile:
                    outfile.write(infile.read())

        #run file to create variables
        subprocess.Popen(['/bin/bash', '-c', "bash " + test_file_name], cwd = slim_dir)

    #setup run
    weight_name = 'pos_%1.9f' % w_pos
    run_name = data_name + '_F_' + fold_id + "_I_" + inner_fold_id + "_fold_" + str(fold_num) + \
               '_W_' + sample_weight_id + '_L_' + hcon_id + '_X_' + xtra_id + "_" + weight_name

    run_dir = slim_dir + "Run/" + data_name + '_F_' + fold_id + '/'
    print 'run_dir: %r' % run_dir
    print 'run_name: %r' % run_name

    run_variables = {
        "comp_name": comp_name,
        "run_dir": run_dir,
        "run_name": run_name,
        "slim_dir": slim_dir,
        "data_dir": data_dir,
        "python_dir": python_dir
    }

    #run script
    script_file = python_dir + script_name

    if execute_script:
        print os.getcwd()
        execfile(script_file, run_variables)
        print os.getcwd()

        #clean up files that were created
        if clean_up_settings_file:
            os.remove(full_test_file_name)
            run_file_pattern = run_dir + run_name + "*.setting"
            for filePath in glob.glob(run_file_pattern):
                if os.path.isfile(filePath):
                    os.remove(filePath)

        if clean_up_results_file:
            results_file_pattern = run_dir + run_name + "_" + results_file_suffix + ".RData"
            for filePath in glob.glob(results_file_pattern):
                if os.path.isfile(filePath):
                    os.remove(filePath)