from .log_loss import *
from .log_loss_weighted import *

try:
    from .fast_log_loss import *
except ImportError:
    print("warning: could not import fast log loss")
    print("warning: returning handle to standard loss functions")
    # todo replace with warning object
    import log_loss as fast_log_loss

try:
    from .lookup_log_loss import *
except ImportError:
    print("warning: could not import lookup log loss")
    print("warning: returning handle to standard loss functions")
    # todo replace with warning object
    import log_loss as lookup_log_loss

