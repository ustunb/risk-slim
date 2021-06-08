import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from scipy.special import expit, logsumexp



class RiskSLIMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, c0_value = 0.01, solver = 'cplex', verbose = True):

        self.c0 = c0_value
        self.coefficient_set = None
        self.solver = solver
        self.verbose = verbose

    def fit(self, X, y):

        # process the data

        # set function handles for loss computation

        # initialization
        # new cuts
        # new bounds
        # new solutions

        # solve
        self.mip = setup_riskslim_mip()

        # store coefficients

        # store solution info in info

        #

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)

        decision = self.decision_function(X)
        decision_2d = np.c_[-decision, decision]
        raise NotImplementedError()

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
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
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))

