"""Tighten loss bounds."""

import numpy as np
from riskslim.data import Bounds


def chained_updates(bounds, C_0_nnz, new_objval_at_feasible = None, new_objval_at_relaxation = None, max_chain_count = 20):
    """Update bounds using chained updates.

    Parameters
    ----------
    bounds : riskslim.data.Bounds
        Parameters bounds.
    C_0_nnz : 1d array
        Regularized coefficients.
    new_objval_at_feasible : float
        New maximum objective value.
    new_objval_at_relaxation : float
        New minimum objective value.
    max_chain_count : int
        Maximum number of times to update chain.

    Returns
    -------
    new_bounds : riskslim.data.Bounds
        Updated parameter bounds.
    """
    new_bounds = Bounds(**bounds.asdict().copy())

    # Update objval_min using new_value (only done once)
    if new_objval_at_relaxation is not None:
        if new_bounds.objval_min < new_objval_at_relaxation:
            new_bounds.objval_min = new_objval_at_relaxation

    # Update objval_max using new_value (only done once)
    if new_objval_at_feasible is not None:
        if new_bounds.objval_max > new_objval_at_feasible:
            new_bounds.objval_max = new_objval_at_feasible

    # We have already converged
    if new_bounds.objval_max <= new_bounds.objval_min:
        new_bounds.objval_max = max(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.objval_min = min(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.loss_max = min(new_bounds.objval_max, new_bounds.loss_max)
        return new_bounds

    # Start update chain
    cnt = 0
    improved = True

    while improved and cnt < max_chain_count:

        improved = False
        L0_penalty_min = np.sum(np.sort(C_0_nnz)[np.arange(int(new_bounds.min_size))])
        L0_penalty_max = np.sum(-np.sort(-C_0_nnz)[np.arange(int(new_bounds.max_size))])

        # loss_min
        if new_bounds.objval_min > L0_penalty_max:
            proposed_loss_min = new_bounds.objval_min - L0_penalty_max
            if proposed_loss_min > new_bounds.loss_min:
                new_bounds.loss_min = proposed_loss_min
                improved = True

        # min_size
        if new_bounds.objval_min > new_bounds.loss_max:
            proposed_min_size = np.ceil((new_bounds.objval_min - new_bounds.loss_max) / np.min(C_0_nnz))
            if proposed_min_size > new_bounds.min_size:
                new_bounds.min_size = proposed_min_size
                improved = True

        # objval_min = max(objval_min, loss_min + L0_penalty_min)
        proposed_objval_min = min(new_bounds.loss_min, L0_penalty_min)
        if proposed_objval_min > new_bounds.objval_min:
            new_bounds.objval_min = proposed_objval_min
            improved = True

        # loss max
        if new_bounds.objval_max > L0_penalty_min:
            proposed_loss_max = new_bounds.objval_max - L0_penalty_min
            if proposed_loss_max < new_bounds.loss_max:
                new_bounds.loss_max = proposed_loss_max
                improved = True

        # max_size
        if new_bounds.objval_max > new_bounds.loss_min:
            proposed_max_size = np.floor((new_bounds.objval_max - new_bounds.loss_min) / np.min(C_0_nnz))
            if proposed_max_size < new_bounds.max_size:
                new_bounds.max_size = proposed_max_size
                improved = True

        # objval_max = min(objval_max, loss_max + penalty_max)
        proposed_objval_max = new_bounds.loss_max + L0_penalty_max
        if proposed_objval_max < new_bounds.objval_max:
            new_bounds.objval_max = proposed_objval_max
            improved = True

        cnt += 1

    return new_bounds


def chained_updates_for_lp(bounds, C_0_nnz, new_objval_at_feasible = None, new_objval_at_relaxation = None, max_chain_count = 20):
    """Update bounds using chained updates for a linear program.

    Parameters
    ----------
    bounds : riskslim.data.Bounds
        Parameters bounds.
    C_0_nnz : 1d array
        Regularized coefficients.
    new_objval_at_feasible : float
        New objective value.
    new_objval_at_relaxation : float
        New objective value relaxing integer requirement.
    max_chain_count : int
        Maximum number of times to update chain.

    Returns
    -------
    new_bounds : riskslim.data.Bounds
        Updated parameter bounds.
    """
    new_bounds = Bounds(**bounds.asdict().copy())

    # Update objval_min using new_value (only done once)
    if new_objval_at_relaxation is not None:
        if new_bounds.objval_min < new_objval_at_relaxation:
            new_bounds.objval_min = new_objval_at_relaxation

    # Update objval_max using new_value (only done once)
    if new_objval_at_feasible is not None:
        if new_bounds.objval_max > new_objval_at_feasible:
            new_bounds.objval_max = new_objval_at_feasible

    if new_bounds.objval_max <= new_bounds.objval_min:
        new_bounds.objval_max = max(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.objval_min = min(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.loss_max = min(new_bounds.objval_max, new_bounds.loss_max)
        return new_bounds

    # start update chain
    chain_count = 0
    improved_bounds = True
    C_0_min = np.min(C_0_nnz)
    C_0_max = np.max(C_0_nnz)
    L0_penalty_min = C_0_min * new_bounds.min_size
    L0_penalty_max = min(C_0_max * new_bounds.max_size, new_bounds.objval_max)

    while improved_bounds and chain_count < max_chain_count:

        improved_bounds = False
        # loss_min
        if new_bounds.objval_min > L0_penalty_max:
            proposed_loss_min = new_bounds.objval_min - L0_penalty_max
            if proposed_loss_min > new_bounds.loss_min:
                new_bounds.loss_min = proposed_loss_min
                improved_bounds = True

        # min_size and L0_penalty_min
        if new_bounds.objval_min > new_bounds.loss_max:
            proposed_min_size = (new_bounds.objval_min - new_bounds.loss_max) / C_0_min
            if proposed_min_size > new_bounds.min_size:
                new_bounds.min_size = proposed_min_size
                L0_penalty_min = max(L0_penalty_min, C_0_min * proposed_min_size)
                improved_bounds = True

        # objval_min = max(objval_min, loss_min + L0_penalty_min)
        proposed_objval_min = min(new_bounds.loss_min, L0_penalty_min)
        if proposed_objval_min > new_bounds.objval_min:
            new_bounds.objval_min = proposed_objval_min
            improved_bounds = True

        # loss max
        if new_bounds.objval_max > L0_penalty_min:
            proposed_loss_max = new_bounds.objval_max - L0_penalty_min
            if proposed_loss_max < new_bounds.loss_max:
                new_bounds.loss_max = proposed_loss_max
                improved_bounds = True

        # max_size and L0_penalty_max
        if new_bounds.objval_max > new_bounds.loss_min:
            proposed_max_size = (new_bounds.objval_max - new_bounds.loss_min) / C_0_min
            if proposed_max_size < new_bounds.max_size:
                new_bounds.max_size = proposed_max_size
                L0_penalty_max = min(L0_penalty_max, C_0_max * proposed_max_size)
                improved_bounds = True

        # objval_max = min(objval_max, loss_max + penalty_max)
        proposed_objval_max = new_bounds.loss_max + L0_penalty_max
        if proposed_objval_max < new_bounds.objval_max:
            new_bounds.objval_max = proposed_objval_max
            L0_penalty_max = min(L0_penalty_max, proposed_objval_max)
            improved_bounds = True

        chain_count += 1

    return new_bounds
