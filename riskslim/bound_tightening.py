import numpy as np


def chained_updates(bounds, C_0_nnz, new_objval_at_feasible = None, new_objval_at_relaxation = None, MAX_CHAIN_COUNT = 20):

    new_bounds = dict(bounds)

    # update objval_min using new_value (only done once)
    if new_objval_at_relaxation is not None:
        if new_bounds['objval_min'] < new_objval_at_relaxation:
            new_bounds['objval_min'] = new_objval_at_relaxation

    # update objval_max using new_value (only done once)
    if new_objval_at_feasible is not None:
        if new_bounds['objval_max'] > new_objval_at_feasible:
            new_bounds['objval_max'] = new_objval_at_feasible

    # we have already converged
    if new_bounds['objval_max'] <= new_bounds['objval_min']:
        new_bounds['objval_max'] = max(new_bounds['objval_max'], new_bounds['objval_min'])
        new_bounds['objval_min'] = min(new_bounds['objval_max'], new_bounds['objval_min'])
        new_bounds['loss_max'] = min(new_bounds['objval_max'], new_bounds['loss_max'])
        return new_bounds

    # start update chain
    chain_count = 0
    improved_bounds = True

    while improved_bounds and chain_count < MAX_CHAIN_COUNT:

        improved_bounds = False
        L0_penalty_min = np.sum(np.sort(C_0_nnz)[np.arange(int(new_bounds['L0_min']))])
        L0_penalty_max = np.sum(-np.sort(-C_0_nnz)[np.arange(int(new_bounds['L0_max']))])

        # loss_min
        if new_bounds['objval_min'] > L0_penalty_max:
            proposed_loss_min = new_bounds['objval_min'] - L0_penalty_max
            if proposed_loss_min > new_bounds['loss_min']:
                new_bounds['loss_min'] = proposed_loss_min
                improved_bounds = True

        # L0_min
        if new_bounds['objval_min'] > new_bounds['loss_max']:
            proposed_L0_min = np.ceil((new_bounds['objval_min'] - new_bounds['loss_max']) / np.min(C_0_nnz))
            if proposed_L0_min > new_bounds['L0_min']:
                new_bounds['L0_min'] = proposed_L0_min
                improved_bounds = True

        # objval_min = max(objval_min, loss_min + L0_penalty_min)
        proposed_objval_min = min(new_bounds['loss_min'], L0_penalty_min)
        if proposed_objval_min > new_bounds['objval_min']:
            new_bounds['objval_min'] = proposed_objval_min
            improved_bounds = True

        # loss max
        if new_bounds['objval_max'] > L0_penalty_min:
            proposed_loss_max = new_bounds['objval_max'] - L0_penalty_min
            if proposed_loss_max < new_bounds['loss_max']:
                new_bounds['loss_max'] = proposed_loss_max
                improved_bounds = True

        # L0_max
        if new_bounds['objval_max'] > new_bounds['loss_min']:
            proposed_L0_max = np.floor((new_bounds['objval_max'] - new_bounds['loss_min']) / np.min(C_0_nnz))
            if proposed_L0_max < new_bounds['L0_max']:
                new_bounds['L0_max'] = proposed_L0_max
                improved_bounds = True

        # objval_max = min(objval_max, loss_max + penalty_max)
        proposed_objval_max = new_bounds['loss_max'] + L0_penalty_max
        if proposed_objval_max < new_bounds['objval_max']:
            new_bounds['objval_max'] = proposed_objval_max
            improved_bounds = True

        chain_count += 1

    return new_bounds


def chained_updates_for_lp(bounds, C_0_nnz, new_objval_at_feasible = None, new_objval_at_relaxation = None, MAX_CHAIN_COUNT = 20):

    new_bounds = dict(bounds)

    # update objval_min using new_value (only done once)
    if new_objval_at_relaxation is not None:
        if new_bounds['objval_min'] < new_objval_at_relaxation:
            new_bounds['objval_min'] = new_objval_at_relaxation

    # update objval_max using new_value (only done once)
    if new_objval_at_feasible is not None:
        if new_bounds['objval_max'] > new_objval_at_feasible:
            new_bounds['objval_max'] = new_objval_at_feasible

    if new_bounds['objval_max'] <= new_bounds['objval_min']:
        new_bounds['objval_max'] = max(new_bounds['objval_max'], new_bounds['objval_min'])
        new_bounds['objval_min'] = min(new_bounds['objval_max'], new_bounds['objval_min'])
        new_bounds['loss_max'] = min(new_bounds['objval_max'], new_bounds['loss_max'])
        return new_bounds

    # start update chain
    chain_count = 0
    improved_bounds = True
    C_0_min = np.min(C_0_nnz)
    C_0_max = np.max(C_0_nnz)
    L0_penalty_min = C_0_min * new_bounds['L0_min']
    L0_penalty_max = min(C_0_max * new_bounds['L0_max'], new_bounds['objval_max'])

    while improved_bounds and chain_count < MAX_CHAIN_COUNT:

        improved_bounds = False
        # loss_min
        if new_bounds['objval_min'] > L0_penalty_max:
            proposed_loss_min = new_bounds['objval_min'] - L0_penalty_max
            if proposed_loss_min > new_bounds['loss_min']:
                new_bounds['loss_min'] = proposed_loss_min
                improved_bounds = True

        # L0_min and L0_penalty_min
        if new_bounds['objval_min'] > new_bounds['loss_max']:
            proposed_L0_min = (new_bounds['objval_min'] - new_bounds['loss_max']) / C_0_min
            if proposed_L0_min > new_bounds['L0_min']:
                new_bounds['L0_min'] = proposed_L0_min
                L0_penalty_min = max(L0_penalty_min, C_0_min * proposed_L0_min)
                improved_bounds = True

        # objval_min = max(objval_min, loss_min + L0_penalty_min)
        proposed_objval_min = min(new_bounds['loss_min'], L0_penalty_min)
        if proposed_objval_min > new_bounds['objval_min']:
            new_bounds['objval_min'] = proposed_objval_min
            improved_bounds = True

        # loss max
        if new_bounds['objval_max'] > L0_penalty_min:
            proposed_loss_max = new_bounds['objval_max'] - L0_penalty_min
            if proposed_loss_max < new_bounds['loss_max']:
                new_bounds['loss_max'] = proposed_loss_max
                improved_bounds = True

        # L0_max and L0_penalty_max
        if new_bounds['objval_max'] > new_bounds['loss_min']:
            proposed_L0_max = (new_bounds['objval_max'] - new_bounds['loss_min']) / C_0_min
            if proposed_L0_max < new_bounds['L0_max']:
                new_bounds['L0_max'] = proposed_L0_max
                L0_penalty_max = min(L0_penalty_max, C_0_max * proposed_L0_max)
                improved_bounds = True

        # objval_max = min(objval_max, loss_max + penalty_max)
        proposed_objval_max = new_bounds['loss_max'] + L0_penalty_max
        if proposed_objval_max < new_bounds['objval_max']:
            new_bounds['objval_max'] = proposed_objval_max
            L0_penalty_max = min(L0_penalty_max, proposed_objval_max)
            improved_bounds = True

        chain_count += 1

    return new_bounds
