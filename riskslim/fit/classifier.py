"""RiskSLIM Classifier."""

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from .optimizer import RiskSLIMOptimizer
from .risk_scores import RiskScores

class RiskSLIMClassifier(RiskSLIMOptimizer, BaseEstimator, ClassifierMixin):
    """RiskSLIM classifier object.

    Parameters
    ----------
    L0_min : int, optional, default: None
        Minimum number of regularized coefficients.
        None defaults to zero.
    L0_max : int, optional, default: None
        Maximum number of regularized coefficients.
        None defaults to length of input variables.
    rho_min : float or 1d array, optional, default: -5.
        Minimum coefficient.
    rho_max : float or 1d array, optional, default: 5.
        Maximum coefficient.
    c0_value : 1d array or float, optional, default: 1e-6
        L0-penalty for all parameters when an integer or for each parameter
        separately when an array.
    max_abs_offset : float, optional, default: None
        Maximum absolute value of intercept. This may be specificed as the first value
        of rho_min and rho_max. However, if rho_min and rho_max are floats, this parameter
        provides a convenient way to set bounds on the offset.
    vtype : str or list of str, optional, default: "I"
        Variable types for coefficients. Must be either "I" for integers or "C" for floats.
    settings, dict, optional, defaults: None
        Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
        None defaults to settings defined in riskslim.defaults.
    coefficient_set : riskslim.coefficient_set.CoefficientSet, optional, default: None
        Contraints (bounds) on coefficients of input variables.
        If None, this is constructed based on values passed to other initalization kwargs.
        If not None, other kwargs may be overwritten.
    verbose : bool, optional, default: True
        Prints out log information if True, supresses if False.
    """

    def __init__(
        self, L0_min=None, L0_max=None, rho_min=-5., rho_max=5., c0_value=1e-6,
        max_abs_offset=None, vtype="I", settings=None, coef_set=None, verbose=True
    ):
        # Initalize super optimizer class
        super().__init__(
            L0_min, L0_max, rho_min, rho_max, c0_value,
            max_abs_offset, vtype, settings, coef_set, verbose
        )


    def fit(self, X, y, variable_names=None, outcome_name=None, sample_weights=None):
        """Fit RiskSLIM classifier.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        variable_names : list of str, optional, default: None
            Names of each features. Only needed if coefficients is not passed on
            initalization. None defaults to generic variable names.
        outcome_name : str, optional, default: None
            Name of the output class.
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        """

        super().optimize(X, y, variable_names, outcome_name, sample_weights)
        self.scores = RiskScores(self)


    def predict(self, X):
        """Predict labels.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.

        Returns
        -------
        y_pred : 1d-array
            Predicted labels of X.
        """
        assert self.fitted
        return np.sign(X.dot(self.coefficients))


    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.

        Returns
        -------
        probs : 1d array
            Probability of classes.
        """
        assert self.fitted
        return expit(X.dot(self.coefficients))


    def predict_log_proba(self, X):
        """Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.y``.
        """
        return np.log(self.predict_proba(X))


    def report(self, file_name=None, show=True):
        """Create a RiskSLIM report using plotly.

        Parameters
        ----------
        file_name : str
            Name of file and extension to save report to.
            Supported extensions include ".pdf" and ".html".
        show : bool, optional, default: True
            Calls fig.show() if True.
        """
        self.scores.report(file_name, show)
