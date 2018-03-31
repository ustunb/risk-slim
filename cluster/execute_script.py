#!/usr/bin/python
if __name__ == '__main__':

    import os
    import sys
    import platform
    print("python version: %s" % platform.python_version())

    comp_name = str(sys.argv[1])
    run_name = str(sys.argv[2])
    run_dir = str(sys.argv[3])
    
    if len(sys.argv) >= 5:
        script_name = str(sys.argv[4])
    else:
        script_name = "onetree_slim_script.py"
    
    if len(sys.argv) >= 6:
        cplex_version = str(sys.argv[5])
    else:
        cplex_version = "12.7"
    
    #setup directories
    slim_dir = os.getcwd() + "/"

    if comp_name == 'berkmac':

        slim_dir = "/Users/berk/Dropbox (MIT)/Research/SLIM/"
        cplex_base_dir = '/home/ustunb/software/CPLEX_Studio' + cplex_version.replace(".", "") + '/cplex/python/'

        if cplex_version == "12.8":
            if sys.version_info.major == 2:
                cplex_dir = cplex_base_dir + '2.7/x86-64_osx/'
            elif sys.version_info.major == 3:
                if sys.version_info.minor <= 5:
                    cplex_dir = cplex_base_dir + '3.5/x86-64_osx/'
                else:
                    cplex_dir = cplex_base_dir + '3.6/x86-64_osx/'
        else:
            if sys.version_info.major == 2:
                cplex_dir = cplex_base_dir + '2.7/x86-64_osx/'
            else:
                cplex_dir = cplex_base_dir + '3.5/x86-64_osx/'

    elif comp_name == 'odyssey':

        slim_dir = "/n/home01/berk/"
        cplex_base_dir = '/home/ustunb/software/CPLEX_Studio' + cplex_version.replace(".", "") + '/cplex/python/'
        if cplex_version >= '12.7':
            cplex_dir = cplex_base_dir + '2.7/x86-64_linux/'
        else:
            cplex_dir = cplex_base_dir + '2.6/x86-64_linux/'

    elif comp_name in ('equity', 'engagement', 'svante'):

        slim_dir = "/home/ustunb/SLIM/"
        cplex_base_dir = '/home/ustunb/software/CPLEX_Studio' + cplex_version.replace(".", "") + '/cplex/python/'
        if cplex_version >= '12.7':
            cplex_dir = cplex_base_dir + '2.7/x86-64_linux/'
        else:
            cplex_dir = cplex_base_dir + '2.6/x86-64_linux/'

    sys.path.insert(0, cplex_dir)
    data_dir = slim_dir + "Data/"
    python_dir = slim_dir + "Python/"

    run_variables = {
        "comp_name": comp_name,
        "run_dir": run_dir,
        "run_name": run_name,
        "slim_dir": slim_dir,
        "data_dir": data_dir,
        "python_dir": python_dir
    }
    script_file = python_dir + script_name

    print("executing file:" + script_file)
    print("comp_name:   " + comp_name)
    print("run_name:    " + run_name)
    print("run_dir:     " + run_dir)

    exec(open(script_file).read(), run_variables)
    print("Finished executing file:" + script_file)
    print("Quitting...")
    sys.exit()
    