from __future__ import print_function
import sys
from setuptools import setup, find_packages

DISTNAME = 'risk-slim'
DESCRIPTION = "optimized risk scores on large-scale datasets, " + \
              "with CPLEX and Python"
AUTHOR = 'Berk Ustun'
AUTHOR_EMAIL = 'ustunb@mit.edu'
URL = 'https://github.com/ustunb/risk-slim'
#LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/ustunb/risk-slim'
VERSION = '0.0.0'


with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setup(name=DISTNAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=INSTALL_REQUIRES,
      #license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      zip_safe=False,
      )