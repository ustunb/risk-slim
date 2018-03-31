#!/usr/bin/python
if __name__ == '__main__':

    import os
    import glob
    import subprocess

    #testing parameters
    script_name = "rounding_methods_script.py"
    clean_up_test_files = True
    clean_up_results_file = True

    #test case parameters
    data_name = "recidivism_v01_rearrest"
    hcon_id = "U000"
    xtra_id = "TEST"
    fold_id = "K05N01"
    inner_fold_id = "NONE"
    fold_num = 0
    sample_weight_id = "NONE"
    w_pos = 1.000
    w_neg = 2.0 - w_pos

    #directories
    comp_name = "berkmac"
    slim_dir = "/Users/berk/Dropbox (MIT)/Research/SLIM/"
    data_dir = slim_dir + "Data/"
    python_dir = slim_dir + "Python/"

    print("testing %s on %s" % (script_name, comp_name))
    print("current directory: %s" % (os.getcwd()))

    #create files
    body_file_name = slim_dir + 'test_body_file.sh'

    #create header file
    header_file_name = slim_dir + 'header_file.sh'
    with open(header_file_name, 'w') as header_file:
        header_file.write("#!/bin/bash\n")
        header_file.write("comp_name='%s'\n" % comp_name)
        header_file.write("data_name='%s'\n" % data_name)
        header_file.write("fold_id='%s'\n" % fold_id)
        header_file.write("inner_fold_id='%s'\n" % inner_fold_id)
        header_file.write("fold_num='%d'\n" % fold_num)
        header_file.write("sample_weight_id='%s'\n" % sample_weight_id)
        header_file.write("hcon_id='%s'\n" % hcon_id)
        header_file.write("xtra_id='%s'\n" % xtra_id)
        header_file.write("w_pos=%1.3f\n" % w_pos)
        header_file.write("w_neg=%1.3f\n" % w_neg)
        header_file.write("script_name='%s'\n" % script_name)
        header_file.write("run_script_name='%s'\n" % script_name)

    #add body to header
    test_file_name = slim_dir + 'test_script.sh'
    with open(test_file_name, 'w') as outfile:
        for fname in [header_file_name, body_file_name]:
            with open(fname) as infile:
                outfile.write(infile.read())

    #run file to create variables
    subprocess.Popen(['/bin/bash', '-c', "bash " + os.path.basename(test_file_name)], cwd = slim_dir)

    #setup run
    weight_name = ('neg_' + '%1.3f' + '_pos_' + '%1.3f') % (w_neg, w_pos)
    run_name = data_name + '_F_' + fold_id + '_L_' + hcon_id + '_X_' + xtra_id + "_" + weight_name
    run_dir = slim_dir + "Run/" + data_name + '_F_' + fold_id + '/'

    run_variables = {
        "comp_name": comp_name,
        "run_dir": run_dir,
        "run_name": run_name,
        "slim_dir": slim_dir,
        "data_dir": data_dir,
        "python_dir": python_dir
    }

    #run script
    print(os.getcwd())
    script_file = python_dir + script_name
    exec(open(script_file).read(), globals())
    print(os.getcwd())

    #clean up files that were created
    if clean_up_test_files:
        os.remove(test_file_name)
        run_file_pattern = run_dir + run_name + "*.setting"
        for filePath in glob.glob(run_file_pattern):
            if os.path.isfile(filePath):
                os.remove(filePath)

    if clean_up_results_file:
        results_file_pattern = run_dir + run_name + "*.RData"
        for filePath in glob.glob(results_file_pattern):
            if os.path.isfile(filePath):
                os.remove(filePath)
