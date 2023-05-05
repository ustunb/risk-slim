"""RiskSLIM Classifier."""

from inspect import signature

import numpy as np
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring

from .optimizer import RiskSLIMOptimizer
from .risk_scores import RiskScores
from .coefficient_set import CoefficientSet

class RiskSLIMClassifier(RiskSLIMOptimizer, BaseEstimator, ClassifierMixin):
    """RiskSLIM classifier object.

    Parameters
    ----------
    min_size : int, optional, default: None
        Minimum number of regularized coefficients.
        None defaults to zero.
    max_size : int, optional, default: None
        Maximum number of regularized coefficients.
        None defaults to length of input variables.
    min_coef : float or 1d array, optional, default: -5.
        Minimum coefficient.
    max_coef : float or 1d array, optional, default: 5.
        Maximum coefficient.
    c0_value : 1d array or float, optional, default: 1e-6
        L0-penalty for all parameters when an integer or for each parameter
        separately when an array.
    max_abs_offset : float, optional, default: None
        Maximum absolute value of intercept. This may be specificed as the first value
        of min_coef and rho_max. However, if min_coef and rho_max are floats, this parameter
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
        self, min_size=None, max_size=None, min_coef=-5., max_coef=5., c0_value=1e-6, max_abs_offset=None, verbose=True, **kwargs):

        #todo: option for variable name, outcome_name

        self.verbose = verbose
        # add
        if 'coef_set' in kwargs:
            assert isinstance(kwargs['coef_set'], CoefficientSet)
            coef_set = kwargs.pop('coef_set')
        else:
            coef_set = CoefficientSet(self.variable_names, lb=min_coef, ub=max_coef, c0=c0_value, vtype='I', print_flag=self.verbose)
            # todo set offset to max_abs_offset in coefficient set

        # pull min_coef/max_coef
        self.coef_set = coef_set
        self.max_abs_offset = max_abs_offset

        # todo: pull values from coefficient set
        #  np.max(coef_set.ub), np.min(coef_set.lb), np.min(coef_set.c0)
        self.rho_min = min_coef
        self.rho_max = max_coef
        self.c0_value = c0_value

        # Settings
        # check that 0 < L0_min < L0_max < d
        self.L0_min = min_size #todo: set to 0 by default
        self.L0_max = max_size #todo: set to len(coef_set) - 1 by default

        # todo drop these if not uses
        self.vtype = "I"

        # filter kwargs to only include keys in RiskOptimizer.
        self.settings = kwargs

        self.cv = None
        self.best_estimator = None


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

        # remove variable_names, outcome_names from this
        self.classes_, _ = np.unique(y, return_inverse=True)

        self.variable_names = variable_names
        self.outcome_name = outcome_name
        self.sample_weights = sample_weights

        # Sci-kit learns requires only attributes directly passed in init signature
        #   to be set in init. Other variables are initalized at fit time
        super().__init__(
            self.L0_min, self.L0_max, self.rho_min, self.rho_max, self.c0_value,
            self.max_abs_offset, self.vtype, self.settings, self.coef_set, self.verbose
        )

        # Fit
        super().optimize(X, y, variable_names, outcome_name, sample_weights)
        self.scores = RiskScores(self, self.X, self.y)


    def fitcv(self, X, y, variable_names=None, outcome_name=None, sample_weights=None, k=5, scoring= "roc_auc"):
        """Validate RiskSLIM classifier.

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
        scoring : str, callable, list, tuple, or dict, default: "roc_auc"
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            - a single string (see sklearn `scoring_parameter`);
            - a callable (see sklearn `scoring`) that returns a single value.

        k : int, sklearn cross-validation generator or an iterable, default: 5
            Determines the cross-validation splitting strategy.
            Possible inputs for k are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

        """

        fit_params = {
            "variable_names": variable_names,
            "outcome_name": outcome_name,
            "sample_weights": sample_weights
        }

        # Run cross-validation
        cv_results = cross_validate(self, X, y, k, return_estimator=True, scoring='roc_auc', fit_params=fit_params)
        # Get scorer and cross-validator
        scorer = check_scoring(self, scoring)
        self.cv = check_cv(cv=k, y=y)

        # Select the estimator that maximizes the scoring method across all folds
        cv_scores = np.zeros((self.cv.n_splits, self.cv.n_splits, 1))
        for i_est, estimator in enumerate(cv_results['estimator']):
            for i_fold, (_, test) in enumerate(self.cv.split(X)):
                cv_scores[i_est, i_fold] = scorer(estimator, X[test], y[test])

        self.best_estimator = cv_results['estimator'][np.argmax(cv_scores.mean(axis=1)[:, 0])]
        self.scores = RiskScores(self.best_estimator, X, y, cv=self.cv)


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
        return expit(X.dot(self.rho))


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


    def decision_function(self, X):
        """Predict confidence scores for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        return X.dot(self.coefficients)


    def create_report(self, file_name=None, show=False):
        """Create a RiskSLIM create_report using plotly.

        Parameters
        ----------
        file_name : str
            Name of file and extension to save create_report to.
            Supported extensions include ".pdf" and ".html".
        show : bool, optional, default: True
            Calls fig.show() if True.
        """
        if show:
            self.scores.report(file_name, show)
        else:
            return self.scores.report(file_name, show)