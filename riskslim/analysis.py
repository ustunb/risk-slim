import warnings
import numpy as np


# ACCURACY STATISTICS COMPUTATION
def get_prediction(x, rho):
    return np.sign(x.dot(rho))


def get_true_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] == 1)


def get_false_positives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] == 1)


def get_true_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[~pos_ind] != 1)


def get_false_negatives_from_pred(yhat, pos_ind):
    return np.sum(yhat[pos_ind] != 1)


def get_accuracy_stats(model, data, error_checking=True):
    accuracy_stats = {
        'train_true_positives': np.nan,
        'train_true_negatives': np.nan,
        'train_false_positives': np.nan,
        'train_false_negatives': np.nan,
        'valid_true_positives': np.nan,
        'valid_true_negatives': np.nan,
        'valid_false_positives': np.nan,
        'valid_false_negatives': np.nan,
        'test_true_positives': np.nan,
        'test_true_negatives': np.nan,
        'test_false_positives': np.nan,
        'test_false_negatives': np.nan,
    }

    model = np.array(model).reshape(data['X'].shape[1], 1)

    # training set
    data_prefix = 'train'
    X_field_name = 'X'
    Y_field_name = 'Y'
    Yhat = get_prediction(data['X'], model)
    pos_ind = data[Y_field_name] == 1

    accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

    if error_checking:
        N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                   accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                   accuracy_stats[data_prefix + '_' + 'false_positives'] +
                   accuracy_stats[data_prefix + '_' + 'false_negatives'])
        assert data[X_field_name].shape[0] == N_check

    # validation set
    data_prefix = 'valid'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_validation_set = (X_field_name in data and
                          Y_field_name in data and
                          data[X_field_name].shape[0] > 0 and
                          data[Y_field_name].shape[0] > 0)

    if has_validation_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    # test set
    data_prefix = 'test'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_test_set = (X_field_name in data and
                    Y_field_name in data and
                    data[X_field_name].shape[0] > 0 and
                    data[Y_field_name].shape[0] > 0)

    if has_test_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    return accuracy_stats


# TODO


def get_calibration_metrics(model, data):

    scores = (data['X'] * data['Y']).dot(model)
    raise NotImplementedError()

    #distinct scores

    #compute calibration error at each score

    full_metrics = {
        'scores': float('nan'),
        'count': float('nan'),
        'predicted_risk': float('nan'),
        'empirical_risk': float('nan')
    }

    cal_error = np.sqrt(np.sum(a*(a-b)^2)) ( - full_metrics['empirical_risk'])

    summary_metrics = {
        'mean_calibration_error': float('nan')
    }

    #counts
    #metrics
    #mean calibration error across all scores

    pass


def get_roc_metrics(model, data):

    raise NotImplementedError()


# ROC Curve + AUC
# adapted from scikit-learn/sklearn/metrics/ranking.py (did not want to import scikit-learn to reduce dependencies)
def get_roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True):
    """Compute Receiver operating characteristic (ROC)
    Note: this implementation is restricted to the binary classification task.
    Read more in the :ref:`User Guide <roc_metrics>`.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    drop_intermediate : boolean, optional (default=True)
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
        .. versionadded:: 0.17
           parameter *drop_intermediate*.
    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].
    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    See also
    --------
    roc_auc_score : Compute Area Under the Curve (AUC) from prediction scores
    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.
    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([ 0. ,  0.5,  0.5,  1. ])
    >>> tpr
    array([ 0.5,  0.5,  1. ,  1. ])
    >>> thresholds
    array([ 0.8 ,  0.4 ,  0.35,  0.1 ])
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, ",
                      "true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    np.check_consistent_length(y_true, y_score)
    y_true = np.column_or_1d(y_true)
    y_score = np.column_or_1d(y_score)
    np.assert_all_finite(y_true)
    np.assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = np.column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = np.stable_cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def auc(x, y, reorder=False):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.
    Parameters
    ----------
    x : array, shape = [n]
        x coordinates.
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default=False)
        If True, assume that the curve is ascending in the case of ties, as for
        an ROC curve. If the curve is non-ascending, the result will be wrong.
    Returns
    -------
    auc : float
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75
    See also
    --------
    roc_auc_score : Computes the area under the ROC curve
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """


    #check_consistent_length(x, y)
    #x = column_or_1d(x)
    #y = column_or_1d(y)

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    if reorder:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area

