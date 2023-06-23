from .version import __version__
from .optimizer import RiskSLIMOptimizer
from .classifier import RiskSLIMClassifier
from .coefficient_set import CoefficientSet
from .utils import load_data_from_csv, print_model

try:
    import cplex
except:
    raise ImportError("CPLEX must be installed.")