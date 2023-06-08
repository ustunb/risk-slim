"""RiskSLIM Classifier."""

from pathlib import Path
import numpy as np
from scipy.special import expit

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, check_cv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import check_scoring

from .optimizer import RiskSLIMOptimizer
from .risk_score import RiskScore
from .coefficient_set import CoefficientSet
from .utils import default_variable_names

class RiskSLIMClassifier(BaseEstimator, ClassifierMixin):
    """RiskSLIM classifier

    Parameters
    ----------
    min_size : int, optional, default: None
        Minimum model size
        None defaults to zero.
    max_size : int, optional, default: None
        Maximum model size.
        None defaults to length of input variables.
    min_coef : numeric or 1d array, optional, default: None
        Minimum coefficient.
        None default to -5
    max_coef : numeric or 1d array, optional, default: None
        Maximum coefficient.
        None defaults to 5.
    c0_value : 1d array or float, optional, default: None
        L0-penalty for all parameters when an integer or for each parameter
        separately when an array.
        None defaults to 1e-6.
    max_abs_offset : float, optional, default: None
        Maximum absolute value of intercept. This may be specificed as the first value
        of min_coef and max_coef. However, if min_coef and max_coef are floats, this parameter
        provides a convenient way to set bounds on the offset.
    variable_names : list of str, optional, default: None
        Names of each features. Only needed if coefficients is not passed on initalization.
        None defaults to generic variable names.
    outcome_name : str, optional, default: None
        Name of the output class.
    verbose : bool, optional, default: True
        Prints out log information if True, supresses if False.
    constraints : list of tuples, optional, deafault: None
        Tuples for adding constraints (name, var_inds, values, sense, rhs).
        The recommended method for adding constraints is via the super classes
        .add_constraint method. This kwarg is included for sklearn object cloning.
    settings : dict, optional, default: None
        Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
        Use of this explicit kwarg allows cloning via sklearn (e.g. for grid search).
        Settings are combined with **kwargs and then parsed. Parameters including in
        in **kwargs will not be accessible or set after a clone.
    **kwargs
        May include key value pairs:

            "coef_set" : riskslim.coefficient_set.CoefficientSet
                Contraints (bounds) on coefficients of input variables.
                If None, this is constructed based on values passed to other initalization kwargs.
                If not None, other kwargs may be overwritten.

            "vtype" : str or list of str
                Variable types for coefficients.
                Must be either "I" for integers or "C" for floats.

            **settings : unpacked dict
                Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
                Defaults are defined in defaults.DEFAULT_LCPA_SETTINGS.

    """

    def __init__(
        self,
        min_size=None,
        max_size=None,
        min_coef=-5,
        max_coef=5,
        c0_value=1e-6,
        max_abs_offset=None,
        variable_names=None,
        outcome_name=None,
        verbose=True,
        constraints=None,
        settings=None,
        **kwargs
    ):

        # Pull min_coef/max_coef
        #   Note: max offset is set in coef_set during .optimize
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.c0_value = c0_value
        self.min_size = min_size
        self.max_size = max_size
        self.max_abs_offset = max_abs_offset

        self.variable_names = variable_names
        self.outcome_name = outcome_name
        self.verbose = verbose

        # Coefficient set
        self.coef_set = kwargs.pop("coef_set", None)
        self.vtype = kwargs.pop("vtype", "I")

        # Settings: verified in super's init call to validate_settings
        #   Must contain keys defined in defaults.py
        self.settings = settings

        if self.settings is None:
            self.settings = {}

        self.settings.update(kwargs)

        self.scores = None
        self.calibrated_estimator = None

        # Cross-validation
        self.cv = None
        self.cv_results = None
        self.cv_calibrated_estimators_ = None

        # Custom constraints
        self.constraints = constraints if constraints is not None else []
        self.fitted = False

        self.optimizer = None

    def __repr__(self):
        if self.fitted:
            return self.scores.__repr__()
        else:
            return super().__repr__()

    def fit(self, X, y, sample_weights=None):
        """Fit RiskSLIM classifier.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        """

        #todo: move data checking into a standalone function that we call in __init__ and here
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        inds = np.less_equal(y, 0)
        if len(inds) > 0:
            y[inds] = -1
        n, d = X.shape
        self.sample_weights = sample_weights
        self.classes_, _ = np.unique(y, return_inverse=True)

        # Create or pop coefficient set from kwargs
        #   This must be done here rather than init, since variable_names is optional
        #   it's length can only be inferred from X.
        if self.variable_names is None:
            self.variable_names = default_variable_names(n_variables = d, include_intercept = True)

        if self.coef_set is None:
            self.coef_set = CoefficientSet(
                    variable_names = self.variable_names,
                    lb=self.min_coef,
                    ub=self.max_coef,
                    c0=self.c0_value,
                    vtype=self.vtype,
                    print_flag=self.verbose
                    )
            self.vtype = self.coef_set.vtype

        assert isinstance(self.coef_set, CoefficientSet)
        self.min_coef = np.min(self.coef_set.lb)
        self.max_coef = np.max(self.coef_set.ub)
        self.c0_value = self.coef_set.c0


        # Default max size is unknown until coefficient set is constructed,
        #   assuming variable_names is not passed during initalization
        self.min_size = 0 if self.min_size is None else self.min_size
        self.max_size = len(self.coef_set) - 1 if self.max_size  is None else self.max_size

        # Sci-kit learns requires only attributes directly passed in init signature
        #   to be set in init. Other variables are initalized at fit time
        self.optimizer = RiskSLIMOptimizer(
            min_size=self.min_size,
            max_size=self.max_size,
            min_coef=self.min_coef,
            max_coef=self.max_coef,
            c0_value=self.c0_value,
            max_abs_offset=self.max_abs_offset,
            variable_names=self.variable_names,
            outcome_name=self.outcome_name,
            verbose=self.verbose,
            # Captured by **kwargs
            coef_set=self.coef_set,
            vtype=self.vtype,
            constraints=self.constraints,
            **self.settings
        )

        # Fit
        self.optimizer.optimize(X, y, self.sample_weights)

        # Attributes
        self.X = X
        self.y = y
        self.rho = self.optimizer.rho
        self.fitted = True

        self.scores = RiskScore(self)

    def fit_cv(
        self,
        X,
        y,
        sample_weights=None,
        k=5,
        scoring="roc_auc",
        n_jobs=1
    ):
        """Train RiskSLIM classifier.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_samples, 1). Must all be positive.
        k : int, sklearn cross-validation generator or an iterable, default: 5
            Determines the cross-validation splitting strategy.
            Possible inputs for k are:

            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.

        scoring : str, callable, list, tuple, or dict, default: "roc_auc"
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            - a single string (see sklearn `scoring_parameter`);
            - a callable (see sklearn `scoring`) that returns a single value.

        n_jobs : int, optional, default: 1
            Number of jobs to run in parallel. -1 defaults to max cores or threads.
        """

        # Get scorer and cross-validator
        scoring = check_scoring(self, scoring)
        self.cv = check_cv(cv=k, y=y)

        # Run cross-validation
        self.cv_results = cross_validate(
            self,
            X=X,
            y=y,
            cv=self.cv,
            return_estimator=True,
            scoring=scoring,
            fit_params={"sample_weights": sample_weights},
            n_jobs=n_jobs
        )

        # Fit an estimator on the entire dataset and compute scores
        if self.scores is None:
            self.fit(X, y, sample_weights)


    def recalibrate(self, X, y, sample_weights=None, method= "sigmoid"):
        """Fit RiskSLIM classifier via Platt Scaling.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        method : {"sigmoid", "isotonic"}
            Linear classifier used to recalibrate scores.
        """
        # todo: call check_data
        # todo: add support for kwargs (method = 'sigmoid') should be

        if not self.fitted:
            raise ValueError("fit RiskSLIM before calling recalibrate")

        if self.cv_results is not None:
            # Compute an ensemble (1 per fold) of calibrators
            self.cv_calibrated_estimators_ = []

            for train, _ in self.cv.split(X, y.reshape(-1)):

                _X = X[train]
                _y = y[train]

                clf = CalibratedClassifierCV(
                    self,
                    cv="prefit",
                    method=method
                )

                clf.fit(_X, _y, sample_weights)
                self.cv_calibrated_estimators_.append(clf)

        # Compute single calibrator if fit_cv was not used
        self.calibrated_estimator = CalibratedClassifierCV(
            self,
            cv="prefit",
            method=method
        )

        self.calibrated_estimator.fit(X, y, sample_weights)

        # Compute scores from the calibrated estimator
        self.scores = RiskScore(self)

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
        if self.calibrated_estimator is None:
            # Normal case
            y_pred = np.sign(X.dot(self.coefficients))
        elif isinstance(self.calibrated_estimator, CalibratedClassifierCV):
            # Calibrator
            y_pred = self.calibrated_estimator.predict(X)

        return y_pred

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

        if self.calibrated_estimator is None:
            # Normal case
            proba = expit(X.dot(self.rho))
        elif isinstance(self.calibrated_estimator, CalibratedClassifierCV):
            # Calibrator
            proba = self.calibrated_estimator.predict_proba(X)[:, 1]

        return proba

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

    def create_report(self, file_name = None, show = True, only_table = False, overwrite = True):
        """Create a RiskSLIM report using plotly.

        Parameters
        ----------
        file_name : str
            Name of file and extension to save report to.
            Supported extensions include ".pdf" and ".html".
        show : bool, optional, default: True
            Calls fig.show() if True.
        """
        # overwrite
        if file_name is None:
            f = None
        else:
            f = Path(file_name)
            if not overwrite:
                assert f.exists(), f'file {file_name} exists'
        self.scores.create_report(file_name = f, show = show, only_table=only_table)
        return f
