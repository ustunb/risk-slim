#!/usr/bin/env python

"""
This script builds loss functions using Cython on a local machine.
It must be launched from the command line using the following command:


python build_cython_loss_functions.py build_ext --inplace


"""

import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import scipy

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



# if __name__ == "__main__":
#     if len(sys.argv) == 0:
#         main()
#     else:
#         main(sys.argv)








