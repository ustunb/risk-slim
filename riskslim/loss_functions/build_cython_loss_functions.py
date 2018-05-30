#!/usr/bin/env python

"""
This script builds loss functions using Cython on a local machine.
To run this script

1. Change to the directory

$REPO_DIR/riskslim/loss_functions

2. Run the following commands in Bash:

python2 build_cython_loss_functions.py build_ext --inplace
python3 build_cython_loss_functions.py build_ext --inplace

"""
import numpy
import scipy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


#fast log loss
ext_modules = [Extension(name = "fast_log_loss",
                         sources=["fast_log_loss.pyx"],
                         include_dirs=[numpy.get_include(), scipy.get_include()],
                         libraries=["m"],
                         extra_compile_args = ["-ffast-math"])]
setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [numpy.get_include(), scipy.get_include()],
    ext_modules = ext_modules,
)

#lookup log loss
ext_modules = [Extension(name = "lookup_log_loss",
                         sources=["lookup_log_loss.pyx"],
                         include_dirs=[numpy.get_include(), scipy.get_include()],
                         libraries=["m"],
                         extra_compile_args = ["-ffast-math"])]

setup(
    cmdclass = {'build_ext': build_ext},
    include_dirs = [numpy.get_include(), scipy.get_include()],
    ext_modules = ext_modules,
)









