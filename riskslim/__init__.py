from .version import __version__
from .optimizer import RiskSLIMOptimizer
from .classifier import RiskSLIMClassifier
from .coefficient_set import CoefficientSet
from .utils import print_model, check_cplex

check_cplex()