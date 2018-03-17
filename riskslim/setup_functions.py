import numpy as np
from .helper_functions import print_log
from .CoefficientSet import CoefficientSet
import riskslim.loss_functions as lossfun



def setup_training_weights(Y, sample_weights = None, w_pos = 1.0, w_neg = 1.0, w_total_target = 2.0):

    """
    Parameters
    ----------
    Y - N x 1 vector with Y = -1,+1
    sample_weights - N x 1 vector
    w_pos - positive scalar showing relative weight on examples where Y = +1
    w_neg - positive scalar showing relative weight on examples where Y = -1

    Returns
    -------
    a vector of N training weights for all points in the training data

    """

    # todo: throw warning if there is no positive/negative point in Y

    # process class weights
    assert w_pos > 0.0, 'w_pos must be strictly positive'
    assert w_neg > 0.0, 'w_neg must be strictly positive'
    assert np.isfinite(w_pos), 'w_pos must be finite'
    assert np.isfinite(w_neg), 'w_neg must be finite'
    w_total = w_pos + w_neg
    w_pos = w_total_target * (w_pos / w_total)
    w_neg = w_total_target * (w_neg / w_total)

    # process case weights
    Y = Y.flatten()
    N = len(Y)
    pos_ind = Y == 1

    if sample_weights is None:
        training_weights = np.ones(N)
    else:
        training_weights = sample_weights.flatten()
        assert len(training_weights) == N
        assert np.all(training_weights >= 0.0)
        #todo: throw warning if any training weights = 0
        #todo: throw warning if there are no effective positive/negative points in Y

    # normalization
    training_weights = N * (training_weights / sum(training_weights))
    training_weights[pos_ind] *= w_pos
    training_weights[~pos_ind] *= w_neg

    return training_weights


def setup_penalty_parameters(coef_set, c0_value = 1e-6):
    """
    get training penalty parameters used internally by LCPA

    Parameters
    ----------

    coef_set - coefficient set (non-empty)
    c0_value - user-specified L0-penalty parameter (must be > 0.0)

    Returns
    -------

    c0_value
    L0_reg_ind
    C_0
    C_0_nnz

    """

    assert c0_value > 0.0, 'default L0_parameter should be positive'
    assert isinstance(coef_set, CoefficientSet)
    c0_value = float(c0_value)
    C_0 = np.array(coef_set.C_0j)
    L0_reg_ind = np.isnan(C_0)
    C_0[L0_reg_ind] = c0_value
    C_0_nnz = C_0[L0_reg_ind]

    return c0_value, C_0, L0_reg_ind, C_0_nnz


def setup_loss_functions(Z, loss_computation = "fast", sample_weights = None, rho_ub = None, rho_lb = None, L0_reg_ind = None, L0_max = None):
    """

    Parameters
    ----------
    Z
    loss_computation
    sample_weights
    rho_ub
    rho_lb
    L0_reg_ind
    L0_max

    Returns
    -------

    """
    # todo fix magic number 20
    # todo load loss based on constraints
    # check if fast/lookup loss is installed

    has_sample_weights = (sample_weights is not None) and (len(np.unique(sample_weights)) > 1)

    if has_sample_weights:

        Z = np.require(Z, requirements=['C'])
        total_sample_weights = np.sum(sample_weights)

        def compute_loss(rho):
            return lossfun.log_loss_weighted.log_loss_value(Z, sample_weights, total_sample_weights, rho)

        def compute_loss_cut(rho):
            return lossfun.log_loss_weighted.log_loss_value_and_slope(Z, sample_weights, total_sample_weights, rho)

        def compute_loss_from_scores(scores):
            return lossfun.log_loss_weighted.log_loss_value_from_scores(sample_weights, total_sample_weights, scores)

        compute_loss_real = compute_loss
        compute_loss_cut_real = compute_loss_cut
        compute_loss_from_scores_real = compute_loss_from_scores

    else:

        has_required_data_for_lookup_table = np.all(Z == np.require(Z, dtype=np.int_)) or (len(np.unique(Z)) <= 20)
        missing_inputs_for_lookup_table = (rho_ub is None) or (rho_lb is None) or (L0_reg_ind is None)

        if loss_computation == 'lookup':
            if missing_inputs_for_lookup_table:
                print_log("MISSING INPUTS FOR LOOKUP LOSS COMPUTATION (rho_lb/rho_ub/L0_reg_ind)")
                print_log("SWITCHING FROM LOOKUP TO FAST LOSS COMPUTATION")
                loss_computation = 'fast'
            elif not has_required_data_for_lookup_table:
                print_log("WRONG DATA TYPE FOR LOOKUP LOSS COMPUTATION (not int or more than 20 distinct values)")
                print_log("SWITCHING FROM LOOKUP TO FAST LOSS COMPUTATION")
                loss_computation = 'fast'
        elif loss_computation == 'fast':
            if has_required_data_for_lookup_table and not missing_inputs_for_lookup_table:
                print_log("SWITCHING FROM FAST TO LOOKUP LOSS COMPUTATION")
                loss_computation = 'lookup'

        if loss_computation == 'fast':
            print_log("USING FAST LOSS COMPUTATION")

            Z = np.require(Z, requirements=['F'])

            def compute_loss(rho):
                return lossfun.fast_log_loss.log_loss_value(Z, rho)

            def compute_loss_cut(rho):
                return lossfun.fast_log_loss.log_loss_value_and_slope(Z, rho)

            def compute_loss_from_scores(scores):
                return lossfun.fast_log_loss.log_loss_value_from_scores(scores)

            compute_loss_real = compute_loss
            compute_loss_cut_real = compute_loss_cut
            compute_loss_from_scores_real = compute_loss_from_scores

        elif loss_computation == 'lookup':

            Z_min, Z_max = np.min(Z, axis=0), np.max(Z, axis=0)
            s_min, s_max = get_score_bounds(Z_min, Z_max, rho_lb, rho_ub, L0_reg_ind, L0_max)

            Z = np.require(Z, requirements=['F'])

            print_log("USING LOOKUP TABLE LOSS COMPUTATION. %d ROWS IN LOOKUP TABLE" % (s_max - s_min + 1))
            loss_value_tbl, prob_value_tbl, tbl_offset = lossfun.lookup_log_loss.get_loss_value_and_prob_tables(s_min, s_max)

            def compute_loss(rho):
                return lossfun.lookup_log_loss.log_loss_value(Z, rho, loss_value_tbl, tbl_offset)

            def compute_loss_cut(rho):
                return lossfun.lookup_log_loss.log_loss_value_and_slope(Z, rho, loss_value_tbl, prob_value_tbl,
                                                                        tbl_offset)

            def compute_loss_from_scores(scores):
                return lossfun.lookup_log_loss.log_loss_value_from_scores(scores, loss_value_tbl, tbl_offset)

            def compute_loss_real(rho):
                return lossfun.fast_log_loss.log_loss_value(Z, rho)

            def compute_loss_cut_real(rho):
                return lossfun.fast_log_loss.log_loss_value_and_slope(Z, rho)

            def compute_loss_from_scores_real(scores):
                return lossfun.fast_log_loss.log_loss_value_from_scores(scores)

        else:

            print_log("USING NORMAL LOSS COMPUTATION")
            Z = np.require(Z, requirements=['C'])

            def compute_loss(rho):
                return lossfun.log_loss.log_loss_value(Z, rho)

            def compute_loss_cut(rho):
                return lossfun.log_loss.log_loss_value_and_slope(Z, rho)

            def compute_loss_from_scores(scores):
                return lossfun.log_loss.log_loss_value_from_scores(scores)

            compute_loss_real = compute_loss
            compute_loss_cut_real = compute_loss_cut
            compute_loss_from_scores_real = compute_loss_from_scores

    return (compute_loss,
            compute_loss_cut,
            compute_loss_from_scores,
            compute_loss_real,
            compute_loss_cut_real,
            compute_loss_from_scores_real)


def setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz):

    get_objval = lambda rho: compute_loss(rho) + np.sum(C_0_nnz * (rho[L0_reg_ind] != 0.0))

    get_L0_norm = lambda rho: np.count_nonzero(rho[L0_reg_ind])

    get_L0_penalty = lambda rho: np.sum(C_0_nnz * (rho[L0_reg_ind] != 0.0))

    get_alpha = lambda rho: np.array(abs(rho[L0_reg_ind]) > 0.0, dtype = np.float_)

    get_L0_penalty_from_alpha = lambda alpha: np.sum(C_0_nnz * alpha)

    return (get_objval, get_L0_norm, get_L0_penalty, get_alpha, get_L0_penalty_from_alpha)


    # Data-Related Computation


def get_conservative_offset(data, coef_set, max_L0_value = None):
    """
    returns a value of the offset that is guaranteed to avoid a loss in performance due to small values. this value is
    overly conservative.

    Parameters
    ----------
    data
    coef_set
    max_L0_value

    Returns
    -------
    optimal_offset = max_abs_score + 1
    where max_abs_score is the largest absolute score that can be achieved using the coefficients in coef_set
    with the training data. note:
    when offset >= optimal_offset, then we predict y = +1 for every example
    when offset <= optimal_offset, then we predict y = -1 for every example
    thus, any feasible model should do better.

    """
    if '(Intercept)' not in coef_set.variable_names:
        raise ValueError("coef_set must contain a variable for the offset called 'Intercept'")

    Z = data['X'] * data['Y']
    Z_min = np.min(Z, axis=0)
    Z_max = np.max(Z, axis=0)

    # get idx of intercept/variables
    offset_idx = coef_set.variable_names.index('(Intercept)')
    variable_idx = [i for i in range(len(coef_set)) if not i == offset_idx]

    # get max # of non-zero coefficients given model size limit
    L0_reg_ind = np.isnan(coef_set.C_0j)[variable_idx]
    trivial_L0_max = np.sum(L0_reg_ind)
    if max_L0_value is not None and max_L0_value > 0:
        max_L0_value = min(trivial_L0_max, max_L0_value)
    else:
        max_L0_value = trivial_L0_max

    # get smallest / largest score
    s_min, s_max = get_score_bounds(Z_min = Z_min[variable_idx],
                                    Z_max = Z_max[variable_idx],
                                    rho_lb = coef_set.lb[variable_idx],
                                    rho_ub = coef_set.ub[variable_idx],
                                    L0_reg_ind = L0_reg_ind,
                                    L0_max = max_L0_value)

    # get max # of non-zero coefficients given model size limit
    conservative_offset = max(abs(s_min), abs(s_max)) + 1
    return conservative_offset


# Score-Based Bounds

def get_score_bounds(Z_min, Z_max, rho_lb, rho_ub, L0_reg_ind = None, L0_max = None):

    edge_values = np.vstack([Z_min * rho_lb, Z_max * rho_lb, Z_min * rho_ub, Z_max * rho_ub])

    if (L0_max is None) or (L0_reg_ind is None) or (L0_max == Z_min.shape[0]):
        s_min = np.sum(np.min(edge_values, axis=0))
        s_max = np.sum(np.max(edge_values, axis=0))
    else:
        min_values = np.min(edge_values, axis=0)
        s_min_reg = np.sum(np.sort(min_values[L0_reg_ind])[0:L0_max])
        s_min_no_reg = np.sum(min_values[~L0_reg_ind])
        s_min = s_min_reg + s_min_no_reg

        max_values = np.max(edge_values, axis=0)
        s_max_reg = np.sum(-np.sort(-max_values[L0_reg_ind])[0:L0_max])
        s_max_no_reg = np.sum(max_values[~L0_reg_ind])
        s_max = s_max_reg + s_max_no_reg

    return s_min, s_max


def get_loss_bounds(Z, rho_ub, rho_lb, L0_reg_ind, L0_max = float('nan')):
    # min value of loss = log(1+exp(-score)) occurs at max score for each point
    # max value of loss = loss(1+exp(-score)) occurs at min score for each point

    rho_lb = np.array(rho_lb)
    rho_ub = np.array(rho_ub)

    # get maximum number of regularized coefficients
    L0_max = Z.shape[0] if np.isnan(L0_max) else L0_max
    num_max_reg_coefs = min(L0_max, sum(L0_reg_ind))

    # calculate the smallest and largest score that can be attained by each point
    scores_at_lb = Z * rho_lb
    scores_at_ub = Z * rho_ub
    max_scores_matrix = np.maximum(scores_at_ub, scores_at_lb)
    min_scores_matrix = np.minimum(scores_at_ub, scores_at_lb)
    assert (np.all(max_scores_matrix >= min_scores_matrix))

    # for each example, compute max sum of scores from top reg coefficients
    max_scores_reg = max_scores_matrix[:, L0_reg_ind]
    max_scores_reg = -np.sort(-max_scores_reg, axis=1)
    max_scores_reg = max_scores_reg[:, 0:num_max_reg_coefs]
    max_score_reg = np.sum(max_scores_reg, axis=1)

    # for each example, compute max sum of scores from no reg coefficients
    max_scores_no_reg = max_scores_matrix[:, ~L0_reg_ind]
    max_score_no_reg = np.sum(max_scores_no_reg, axis=1)

    # max score for each example
    max_score = max_score_reg + max_score_no_reg

    # for each example, compute min sum of scores from top reg coefficients
    min_scores_reg = min_scores_matrix[:, L0_reg_ind]
    min_scores_reg = np.sort(min_scores_reg, axis=1)
    min_scores_reg = min_scores_reg[:, 0:num_max_reg_coefs]
    min_score_reg = np.sum(min_scores_reg, axis=1)

    # for each example, compute min sum of scores from no reg coefficients
    min_scores_no_reg = min_scores_matrix[:, ~L0_reg_ind]
    min_score_no_reg = np.sum(min_scores_no_reg, axis=1)

    min_score = min_score_reg + min_score_no_reg
    assert (np.all(max_score >= min_score))

    # compute min loss
    idx = max_score > 0
    min_loss = np.empty_like(max_score)
    min_loss[idx] = np.log1p(np.exp(-max_score[idx]))
    min_loss[~idx] = np.log1p(np.exp(max_score[~idx])) - max_score[~idx]
    min_loss = min_loss.mean()

    # compute max loss
    idx = min_score > 0
    max_loss = np.empty_like(min_score)
    max_loss[idx] = np.log1p(np.exp(-min_score[idx]))
    max_loss[~idx] = np.log1p(np.exp(min_score[~idx])) - min_score[~idx]
    max_loss = max_loss.mean()

    return min_loss, max_loss
