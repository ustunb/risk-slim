# THIS IS A FILE CONTAINS CODE TO RUN THE ONE TREE CUTTING PLANE ALGORITHM
# IT INCLUDES SETTINGS FOR ALL POSSIBLE HEURISTICS AND WARMUP PROCEDURES
# IT MUST BE CALLED USING execute_script.py

#DEBUGGING
#to debug quickly: call this script using test_onetree_script
global run_name, run_dir, data_dir
import numpy as np


#### MODULES
from riskslim.helper_functions import print_log
from cluster_helper_functions import *
import pandas as pd
from pprint import pprint
from rpy2.robjects import pandas2ri, r as rcmd
from cplex import infinity as CPX_INFINITY
from cplex.callbacks import LazyConstraintCallback, HeuristicCallback #UserCutCallback, IncumbentCallback
from cplex.exceptions import CplexError

#### PRINT RUN INFORMATION
print_log('RUNNING FILE: basic_risk_slim_script.py')
print_log('run_name: %s' % run_name)
print_log('run_dir: %s' % run_dir)
print_log('data_dir: %s' % data_dir)
print_log('BLAS LINKAGE')
np.__config__.show()

#### LOAD HELPER FUNCTIONS
print_log('LOADING FUNCTIONS')

def load_from_disk(value_name, default_value = None, load_dir = run_dir, load_header = run_name):
    loaded_value = load_or_set_default(value_name, default_value, load_dir, load_header, print_flag = True)
    return loaded_value

def print_control(control, rho = None, objval = None):
    print_log("cuts = %d \t UB = %.5f \t LB = %.5f \t GAP = %.1f%%" % (
        control['n_cuts'], control['upperbound'], control['lowerbound'], 100.0 * control['relative_gap']))
    if not np.any(np.isnan(control['incumbent'])):
        print_log('incumbent rho %1.4f %s' % (control['upperbound'], get_rho_string(control['incumbent'])))
    if rho is not None:
        print_log('proposed rho  %1.4f %s' % (objval, get_rho_string(rho)))
    return

def print_results(control, settings):
    print "\n" * 5
    print "rho: (%s)" % get_rho_string(control['incumbent'])
    print "status: %s" % control['cplex_status']
    print "stop reason: " + control['stop_reason'] + "\n"
    print "mipgap: %1.4f%%" % (100.0 * control['relative_gap'])
    print "UB: %1.6f" % control['upperbound']
    print "LB: %1.6f" % control['lowerbound']

    print "\n"
    print "L0_min: %d" % control['bounds']['L0_min']
    print "L0_max: %d" % control['bounds']['L0_max']
    print "loss_min: %1.6f" % control['bounds']['loss_min']
    print "loss_max: %1.6f" % control['bounds']['loss_max']

    print "\n"
    print "total run time: %1.2f seconds" % control['total_run_time']
    print "total cut time: %1.2f seconds" % control['total_cut_time']
    print "total heuristic time: %1.2f seconds" % (
        control['total_round_time'] + control['total_round_then_polish_time'] + control['total_polish_time'])

    print "nodes processed: %d" % control['nodes_processed']
    print "nodes remaining: %d" % control['nodes_remaining']
    print "cuts added: %d" % control['n_cuts']
    print "incumbent updates: %d" % control['n_incumbent_updates']
    print "heuristic updates: %d" % (control['n_heuristic_updates_from_rounding'] + \
                                     control['n_heuristic_updates_from_polishing'] + \
                                     control['n_heuristic_updates_from_rounding_then_polishing'])
    if settings['round_flag']:
        print "\n" * 2
        print "ROUNDING SUMMARY"
        if control['n_rounded'] == 0:
            print "no solutions rounded"
        else:
            print "updates / processed: %d / %d " % (control['n_heuristic_updates_from_rounding'], control['n_rounded'])
            print "total / average time: %1.4f, %1.4f seconds" % (
                control['total_round_time'], control['total_round_time'] / control['n_rounded'])

    if settings["polish_rounded_solutions"]:
        print "\n" * 2
        print "ROUNDING THEN POLISHING SUMMARY"
        if control['n_rounded_then_polished'] ==0:
            print "no solutions rounded then polished"
        else:
            print "updates / processed: %d / %d" % (
                control['n_heuristic_updates_from_rounding_then_polishing'], control['n_rounded_then_polished'])
            print "total / average time: %1.4f, %1.4f seconds" % (control['total_round_then_polish_time'],
                                                                  control['total_round_then_polish_time'] / control['n_rounded_then_polished'])

    if settings["polish_flag"]:
        print "\n" * 2
        print "POLISHING SUMMARY"
        if control['n_polished'] == 0:
            print "no solutions polished"
        else:
            print "updates / processed: %d / %d" % (
                control['n_heuristic_updates_from_polishing'], control['n_polished'])
            print "total / average time: %1.4f, %1.4f seconds" % (
                control['total_polish_time'], control['total_polish_time'] / control['n_polished'])

    if settings['global_bound_flag']:
        print "\n" * 2
        print "BOUNDS SUMMARY"
        print 'global bound updates: %d' % control['n_bound_updates']
        print "global L0_min updates: %d" % control['n_global_cuts_L0_min']
        print "global L0_max updates: %d" % control['n_global_cuts_L0_max']
        print "global loss_min updates: %d" % control['n_global_cuts_loss_min']
        print "global loss_max updates: %d" % control['n_global_cuts_loss_max']
        print "global objval_min updates: %d" % control['n_global_cuts_objval_min']
        print "global objval_max updates: %d" % control['n_global_cuts_objval_max']
    print "\n" * 5

#### READ SETTINGS FROM DISK
print_log('READING SETTINGS FROM DISK')

comp_name = load_from_disk("comp_name", "berkmac")
comp_ram = load_from_disk('comp_ram', 48 * 1024 ** 2)
n_cores = load_from_disk('n_cores', 1)
data_name = load_from_disk("data_name", "breastcancer")
xtra_id = load_from_disk("xtra_id", "1TREE")
fold_id = load_from_disk("fold_id", "K10N01")
inner_fold_id = load_from_disk("inner_fold_id", "NONE")
fold_num = load_from_disk("fold_num", 0)
w_pos = load_from_disk("w_pos", 1.00)
sample_weight_id = load_from_disk("sample_weight_id", "NONE")
c0_value = load_from_disk('c0_value', 0.01)
use_custom_coefficient_set = load_from_disk("use_custom_coefficient_set", False)
loss_computation = load_from_disk('loss_computation', 'fast')
results_file_suffix = load_from_disk("results_file_suffix", "results_raw")
save_results_as_rdata = load_from_disk("save_results_as_rdata", True)

default_cplex_parameters = {
    'randomseed': 0,
    'repairtries': 20,
    'mipemphasis': 0,
    'nodefilesize': (120 * 1024) / n_cores,
    'poolsize': 100,
    'poolrelgap': np.nan,
    'poolreplace': 2,
    'n_cores': n_cores,
    'mipgap': np.finfo('float').eps,
    'absmipgap': np.finfo('float').eps,
    'integrality_tolerance': np.finfo('float').eps,
}
cplex_parameters = {key: load_from_disk("cplex_" + key, default_cplex_parameters[key]) for key in default_cplex_parameters}

default_settings = {
    'max_runtime': 300.0,
    'max_tolerance': 0.000001,
    'max_iterations': 100000,
    #
    'store_stats': True,
    'display_cplex_progress': True,
    'display_updates': False,
    'display_updates_at_rounding': False,
    'display_updates_at_polishing': False,
    #
    'tight_formulation': True,
    'polish_flag': True,
    'round_flag': False,
    'polish_rounded_solutions': True,
    'add_cuts_at_heuristic_solutions': True,
    'global_bound_flag': True,
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
settings = {key: load_from_disk(key, default_settings[key]) for key in default_settings}

default_warmstart_settings = {
    'use_cvx': True,
    'cvx_loss_computation': loss_computation,
    'cvx_display_progress': True,
    'cvx_display_cplex_progress': False,
    'cvx_save_progress': True,
    #
    'cvx_max_runtime': 300.0,
    'cvx_max_runtime_per_iteration': 300.0,
    'cvx_max_cplex_time_per_iteration': 10.0,
    'cvx_max_iterations': 10000,
    'cvx_max_tolerance': 0.0001,
    #
    'cvx_use_sequential_rounding': True,
    'cvx_sequential_rounding_max_runtime': 30.0,
    'cvx_sequential_rounding_max_solutions': 5,
    #
    'use_ntree': True,
    'ntree_loss_computation': loss_computation,
    'ntree_display_progress': True,
    'ntree_display_cplex_progress': False,
    'ntree_save_progress': True,
    #
    'ntree_max_runtime_per_iteration': 10.0,
    'ntree_max_cplex_time_per_iteration': 10.0,
    'ntree_max_runtime': settings['max_runtime'],
    'ntree_max_iterations': settings['max_iterations'],
    'ntree_max_tolerance': settings['max_tolerance'],
    #
    'polishing_after': True,
    'polishing_max_runtime': 30.0,
    'polishing_max_solutions': 5,
}
warmstart_settings = {key: load_from_disk(key, default_warmstart_settings[key]) for key in default_warmstart_settings}


#### INITIALIZE DATA / WEIGHTS / MODEL RELATED OBJECTS

data = load_matlab_data(data_file_name = data_dir + data_name + "_processed.mat",
                        fold_id = fold_id,
                        fold_num = fold_num,
                        inner_fold_id = inner_fold_id,
                        sample_weight_id = sample_weight_id)

hard_constraints = load_hard_constraints(data,
                                         data_file_name = data_dir + data_name + "_processed.mat",
                                         hcon_id = load_from_disk("hcon_id", "U001"),
                                         use_custom_coefficient_set = load_from_disk("use_custom_coefficient_set", False),
                                         max_coefficient = load_from_disk("max_coefficient", 5),
                                         max_offset = load_from_disk("max_offset", 50),
                                         max_L0_value = load_from_disk("max_L0_value", -1))

#TODO: adapt to LATTICE CPA

#TODO MAKE SURE WE ARE STORING ALL OF THE FOLLOWING QUANTITIES
#TODO: AT LEAST WHAT IS USED BY R

script_df = update_script_df('final', new_bounds = control['bounds'], new_stats = script_stats, new_solution = control['incumbent'])

#update computational stats
# if settings['store_stats']:
#     stats['event_id'].append(3)
#     #
#     stats['incumbent_update'].append(incumbent_update_at_termination)
#     stats['heuristic_update'].append(False)
#     stats['heuristic_objval'].append(np.nan)
#     stats['heuristic_update_source'].append(np.nan)
#     #
#     stats['relative_gap'].append(control['relative_gap'])
#     stats['upperbound'].append(control['upperbound'])
#     stats['lowerbound'].append(control['lowerbound'])
#     stats['L0_min'].append(control['bounds']['L0_min'])
#     stats['L0_max'].append(control['bounds']['L0_max'])
#     stats['loss_min'].append(control['bounds']['loss_min'])
#     stats['loss_max'].append(control['bounds']['loss_max'])
#     #
#     stats['n_cuts'].append(0)
#     stats['n_rounded'].append(0)
#     stats['n_polished'].append(0)
#     stats['n_rounded_then_polished'].append(0)
#     stats['n_bound_updates'].append(control['global_bounds_times_used'])
#     #
#     stats['cut_time'].append(0.0)
#     stats['polish_time'].append(0.0)
#     stats['round_time'].append(0.0)
#     stats['round_then_polish_time'].append(0.0)
#     #
#     stats['nodes_processed'].append(control['nodes_processed'])
#     stats['nodes_remaining'].append(control['nodes_remaining'])
#     stats['loss_callback_time'].append(0.0)
#     stats['local_callback_time'].append(0.0)
#     stats['heuristic_callback_time'].append(0.0)
#     stats['total_run_time'].append(control['total_run_time'])


#### POST PROCESSING

#convert stats into results object
if settings['store_stats']:

    results = pd.DataFrame(stats)
    results['event_name'] = 'none'
    results.loc[results['event_id'] == 0, 'event_name'] = 'loss_cb'
    results.loc[results['event_id'] == 1, 'event_name'] = 'heuristic_cb'
    results.loc[results['event_id'] == 2, 'event_name'] = 'local_cb'
    results.loc[results['event_id'] == 3, 'event_name'] = 'final'

    # compute solver time
    results['callback_time'] = results['heuristic_callback_time'] + results['loss_callback_time'] + results['local_callback_time']
    results['total_callback_time'] = pd.Series(results['callback_time']).cumsum()
    results['total_solver_time'] = results['total_run_time'] - results['total_callback_time']
    results['solver_time'] = results['total_solver_time'].diff()
    results.loc[0, 'solver_time'] = 0.0

    # compute other aggregate timings
    results['data_time'] = results['cut_time'] + results['round_time'] + results['polish_time'] + results['round_then_polish_time']
    results['total_cut_time'] = pd.Series(results['cut_time']).cumsum()
    results['total_round_time'] = pd.Series(results['round_time']).cumsum()
    results['total_polish_time'] = pd.Series(results['polish_time']).cumsum()
    results['total_round_then_polish_time'] = pd.Series(results['round_then_polish_time']).cumsum()
    results['total_data_time'] = pd.Series(results['data_time']).cumsum()

    # counts
    results['heuristic_source'] = 'none'
    results.loc[results['heuristic_update_source'] == 1, 'heuristic_source'] = 'rounding'
    results.loc[results['heuristic_update_source'] == 2, 'heuristic_source'] = 'polishing'
    results.loc[results['heuristic_update_source'] == 3, 'heuristic_source'] = 'rounding_then_polishing'
    results['n_rounded'] = pd.Series(results['heuristic_update_source'] == 1).cumsum()
    results['n_polished'] = pd.Series(results['heuristic_update_source'] == 2).cumsum()
    results['n_rounded_then_polished'] = pd.Series(results['n_rounded_then_polished'] == 3).cumsum()
    results['n_cuts_added'] = results['n_cuts']
    results['n_cuts'] = pd.Series(results['n_cuts_added']).cumsum()

    # UNCOMMENT TO COMPUTE ACCURACY STATS FOR EACH MODEL
    # Currently done in AggregateRSCV
    # data = load_matlab_data(data_file_name = data_file_name,
    #                         fold_id = fold_id,
    #                         fold_num = fold_num,
    #                         inner_fold_id = inner_fold_id,
    #                         sample_weight_id = sample_weight_id)

    # model processing
    model_stats = {
        'type': [],
        'model_id': [],
        # UNCOMMENT TO COMPUTE ACCURACY STATS FOR EACH MODEL
        #'L0_norm': [],
        #'train_true_positives': [],
        #'train_true_negatives': [],
        #'train_false_positives': [],
        #'train_false_negatives': [],
        #'valid_true_positives': [],
        #'valid_true_negatives': [],
        #'valid_false_positives': [],
        #'valid_false_negatives': [],
        #'test_true_positives': [],
        #'test_true_negatives': [],
        #'test_false_positives': [],
        #'test_false_negatives': [],
    }

    results['incumbent_model_id'] = pd.Series(results['incumbent_update']).cumsum()  # n_incumbent_updates
    results['heuristic_model_id'] = pd.Series(results['heuristic_update']).cumsum()  # n_heuristic_updates
    results.loc[results['incumbent_model_id'] == 0, 'incumbent_model_id'] = np.nan
    results.loc[results['heuristic_model_id'] == 0, 'heuristic_model_id'] = np.nan

    max_incumbent_model_id = int(results['incumbent_model_id'].max())
    results['heuristic_model_id'] = results['heuristic_model_id'] + max_incumbent_model_id

    model_id_counter = 1
    models = incumbent_solutions + heuristic_solutions

    for i in range(0, len(incumbent_solutions)):

        rho = incumbent_solutions[i]
        model_stats['type'].append('incumbent')
        model_stats['model_id'].append(model_id_counter)

        # UNCOMMENT TO COMPUTE ACCURACY STATS FOR EACH MODEL
        # model_stats['L0_norm'].append(get_L0_norm(rho))
        #
        # accuracy_stats = get_accuracy_stats(rho, data)
        # for key in accuracy_stats.keys():
        #     model_stats[key].append(accuracy_stats[key])

        model_id_counter += 1

    model_id_counter = max_incumbent_model_id + 1

    for i in range(0, len(heuristic_solutions)):

        rho = heuristic_solutions[i]
        model_stats['type'].append('heuristic')
        model_stats['model_id'].append(model_id_counter)
        # UNCOMMENT TO COMPUTE ACCURACY STATS FOR EACH MODEL
        # model_stats['L0_norm'].append(get_L0_norm(rho))
        # accuracy_stats = get_accuracy_stats(rho, data)
        # for key in accuracy_stats.keys():
        #     model_stats[key].append(accuracy_stats[key])
        #
        model_id_counter += 1

pd.Index(script_df['event_name']).get_loc('before_1tree')
script_tail = pd.DataFrame(script_df).tail(1).to_dict(orient = 'records')[0]
script_info = {(key + '_at_start'): script_tail[key] for key in script_tail if key is not 'solution'}

#info dict
info = {
    'comp_name': comp_name,
    'date': time.strftime("%d/%m/%y", time.localtime()),
    'data_name': data_name,
    'hcon_id': hard_constraints['hcon_id'],
    'xtra_id': xtra_id,
    'fold': fold_num,
    'fold_id': fold_id,
    'inner_fold_id': inner_fold_id,
    'sample_weight_id': sample_weight_id,
    'w_pos': w_pos,
    'w_neg': w_neg,
    'C_0': c0_value,
    'smallest_C0_value_flag': smallest_C0_value_flag,
    'use_custom_coefficient_set': use_custom_coefficient_set,
    'max_coefficient': hard_constraints['max_coefficient'],
    'max_offset': hard_constraints['max_offset'],
    'max_L0_value': hard_constraints['max_L0_value'],
    'loss_computation': loss_computation,
    'warmstart_flag': warmstart_settings['use_cvx'] or warmstart_settings['use_ntree'],
}
info.update(control)
info.update(settings_before_solve)
info.update(script_info)
info.update(warmstart_settings)
info.update(info['bounds'])
info.pop('bounds', None)
info.pop('incumbent', None)

print_log("INFO")
pprint(info)

#results
print_log("RESULTS SUMMARY")
print_results(control, settings_before_solve)


#### Convert and Save in R Format
print_log("CONVERTING RESULTS TO DATA FRAMES")
info_df = pd.DataFrame(info, [1])
mip_model_df = pd.DataFrame(control['incumbent'].reshape(1, P), columns = variable_names)

script_df = pd.DataFrame(script_df)
script_models_df = pd.DataFrame(script_df['solution'].tolist(), columns = variable_names)
script_df = script_df.drop('solution', axis = 'columns')

# Detailed Stats for Main Run
results_df = pd.DataFrame([])
model_stats_df = pd.DataFrame([])
models_df = pd.DataFrame([])
if settings['store_stats']:
    results_df = pd.DataFrame(results)
    models_df = pd.DataFrame(models, columns= variable_names)
    model_stats_df = pd.DataFrame(model_stats)


# Detailed Stats for CVX Warmstart
cvx_stats_df = pd.DataFrame([])
cvx_models_df = pd.DataFrame([])
cvx_settings_df = pd.DataFrame([])
if warmstart_settings['use_cvx']:
    cvx_stats_df = pd.DataFrame({key: cvx_stats[key] for key in cvx_stats.keys() if type(cvx_stats[key]) is list and len(cvx_stats[key]) > 1})
    cvx_stats_df.drop(labels = 'solutions', axis = 1, inplace = True)
    cvx_models_df = pd.DataFrame(cvx_stats['solutions'], columns = variable_names)
    cvx_settings_df = pd.DataFrame(cvx_settings, [1])

# Detailed Stats for NTREE Warmstart
ntree_stats_df = pd.DataFrame([])
ntree_models_df = pd.DataFrame([])
ntree_settings_df = pd.DataFrame([])
if warmstart_settings['use_ntree']:
    ntree_stats_df = pd.DataFrame({key: ntree_stats[key] for key in ntree_stats.keys() if
                                   type(ntree_stats[key]) is list and len(ntree_stats[key]) > 1})
    ntree_stats_df.drop(labels = 'solutions', axis = 1, inplace = True)
    ntree_models_df = pd.DataFrame(ntree_stats['solutions'], columns = variable_names)
    ntree_settings_df = pd.DataFrame(ntree_settings, [1])

print_log("FINISHED CONVERTING RESULTS TO DATA FRAMES")

# Save Results in R
results_file_R = run_dir + run_name + "_" + results_file_suffix + ".RData"

try:
    pandas2ri.activate()
    rcmd.assign("info", info_df)
    rcmd.assign("mip_model", mip_model_df)
    rcmd.assign("script", script_df)
    rcmd.assign("script_models", script_models_df)
    rcmd.assign("results", results_df)
    rcmd.assign("models", models_df)
    rcmd.assign("model_stats", model_stats_df)
    rcmd.assign("cvx_stats", cvx_stats_df)
    rcmd.assign("cvx_models", cvx_models_df)
    rcmd.assign("cvx_settings", cvx_settings_df)
    rcmd.assign("ntree_stats", ntree_stats_df)
    rcmd.assign("ntree_models", ntree_models_df)
    rcmd.assign("ntree_settings", ntree_settings_df)
    print_log("SAVING RESULTS TO FILE %s" % results_file_R)
    rcmd.save("info", "mip_model",
              "script", "script_models",
              "results", "models", "model_stats",
              "cvx_stats", "cvx_models", "cvx_settings",
              "ntree_stats","ntree_models", "ntree_settings",
              file = results_file_R)
except:
    print_log("FAILED TO SAVE RESULTS")

print_log("REACHED END OF onetree_risk_slim_script")