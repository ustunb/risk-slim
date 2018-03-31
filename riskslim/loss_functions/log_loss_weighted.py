import numpy as np

def log_loss_value(Z, weights, total_weights, rho):
    """
    computes the value and slope of the logistic loss in a numerically stable way
    supports sample non-negative weights for each example in the training data
    see http://stackoverflow.com/questions/20085768/

    Parameters
    ----------
    Z               numpy.array containing training data with shape = (n_rows, n_cols)
    rho             numpy.array of coefficients with shape = (n_cols,)
    total_weights   numpy.sum(total_weights) (only included to reduce computation)
    weights         numpy.array of sample weights with shape (n_rows,)

    Returns
    -------
    loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*rho))

    """
    scores = Z.dot(rho)
    pos_idx = scores > 0
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(np.exp(-scores[pos_idx]))
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(np.exp(scores[~pos_idx]))
    loss_value = loss_value.dot(weights) / total_weights
    return loss_value

def log_loss_value_and_slope(Z, weights, total_weights, rho):
    """
    computes the value and slope of the logistic loss in a numerically stable way
    supports sample non-negative weights for each example in the training data
    this function should only be used when generating cuts in cutting-plane algorithms
    (computing both the value and the slope at the same time is slightly cheaper)

    see http://stackoverflow.com/questions/20085768/

    Parameters
    ----------
    Z               numpy.array containing training data with shape = (n_rows, n_cols)
    rho             numpy.array of coefficients with shape = (n_cols,)
    total_weights   numpy.sum(total_weights) (only included to reduce computation)
    weights         numpy.array of sample weights with shape (n_rows,)

    Returns
    -------
    loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*rho))
    loss_slope: (n_cols x 1) vector = 1/n_rows * sum(-Z*rho ./ (1+exp(-Z*rho))

    """
    scores = Z.dot(rho)
    pos_idx = scores > 0
    exp_scores_pos = np.exp(-scores[pos_idx])
    exp_scores_neg = np.exp(scores[~pos_idx])

    #compute loss value
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(exp_scores_pos)
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(exp_scores_neg)
    loss_value = loss_value.dot(weights) / total_weights

    #compute loss slope
    log_probs = np.empty_like(scores)
    log_probs[pos_idx] = 1.0 / (1.0 + exp_scores_pos)
    log_probs[~pos_idx] = (exp_scores_neg / (1.0 + exp_scores_neg))
    log_probs -= 1.0
    log_probs *= weights
    loss_slope = Z.T.dot(log_probs) / total_weights

    return loss_value, loss_slope

def log_loss_value_from_scores(weights, total_weights, scores):
    """
    computes the logistic loss value from a vector of scores in a numerically stable way
    where scores = Z.dot(rho)

    see also: http://stackoverflow.com/questions/20085768/

    this function is used for heuristics (discrete_descent, sequential_rounding).
    to save computation when running the heuristics, we store the scores and
    call this function to compute the loss directly from the scores
    this reduces the need to recompute the dot product.

    Parameters
    ----------
    scores          numpy.array of scores = Z.dot(rho)
    total_weights   numpy.sum(total_weights) (only included to reduce computation)
    weights         numpy.array of sample weights with shape (n_rows,)

    Returns
    -------
    loss_value  scalar = 1/n_rows * sum(log( 1 .+ exp(-Z*rho))

    """
    pos_idx = scores > 0
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(np.exp(-scores[pos_idx]))
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(np.exp(scores[~pos_idx]))
    loss_value = loss_value.dot(weights) / total_weights

    return loss_value