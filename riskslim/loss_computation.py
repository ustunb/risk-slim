import numpy as np
from .bounds import get_score_bounds


def get_loss_functions(data, coef_set, loss_computation, max_size = None):
    """Initalize loss functions."""

    # todo: set default value for loss computation (should try to use lookup, fast and default to normal otherwise)

    assert loss_computation in ("normal", "fast", "lookup")
    use_lookup_table = data._integer_data
    max_size = data.d if max_size is None else max_size


    if loss_computation == "lookup":

        assert data.is_integer_data
        from .loss_functions import lookup_log_loss as lf
        data._Z = np.require(data._Z, requirements=["F"], dtype=np.float64)

        s_min, s_max = get_score_bounds(
                Z_min=np.min(data.Z, axis=0),
                Z_max=np.max(data.Z, axis=0),
                rho_lb=coef_set.lb,
                rho_ub=coef_set.ub,
                L0_reg_ind=coef_set.penalized_indices(),
                max_size=max_size,
                )

        loss_value_tbl, prob_value_tbl, tbl_offset = lf.get_loss_value_and_prob_tables(s_min, s_max)

        from .loss_functions import fast_log_loss as lfr
        handles = {
            'loss': lambda rho: lf.log_loss_value(data.Z, rho, loss_value_tbl, tbl_offset),
            'loss_cut': lambda rho: lf.log_loss_value_and_slope(data.Z, rho, loss_value_tbl, prob_value_tbl, tbl_offset),
            'loss_from_scores': lambda scores: lf.log_loss_value_from_scores(scores, loss_value_tbl, tbl_offset),
            #
            'loss_real': lambda rho: lfr.log_loss_value(data.Z, rho),
            'loss_cut_real': lambda rho: lfr.log_loss_value_and_slope(data.Z, rho),
            'loss_from_scores_real': lambda scores: lfr.log_loss_value_from_scores(scores)
            }

    if loss_computation == "fast":

        from .loss_functions import fast_log_loss as lf
        data._Z = np.require(data._Z, requirements=["F"])
        handles = {
            'loss': lambda rho: lf.log_loss_value(data.Z, np.asfortranarray(rho)),
            'loss_cut': lambda rho: lf.log_loss_value_and_slope(data.Z, np.asfortranarray(rho)),
            'loss_from_scores': lambda scores: lf.log_loss_value_from_scores(scores),
            }

        # add handles for real-valued losses
        handles_real = {k + '_real': f for k, f in handles.items()}
        handles.update(handles_real)

    if loss_computation == "normal":

        from .loss_functions import log_loss as lf
        data._Z = np.require(data._Z, requirements=["C"])

        handles = {
            'loss': lambda rho: lf.log_loss_value(data.Z, rho),
            'loss_cut': lambda rho: lf.log_loss_value_and_slope(data.Z, rho),
            'loss_from_scores': lambda scores: lf.log_loss_value_from_scores(scores),
            }

        # add handles for real-valued losses
        handles_real = {k + '_real': f for k, f in handles.items()}
        handles.update(handles_real)

    return handles