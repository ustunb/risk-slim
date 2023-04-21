"""Bound tightening tests."""

import pytest
import numpy as np
from riskslim.data import Bounds
from riskslim.bound_tightening import chained_updates, chained_updates_for_lp


@pytest.mark.parametrize('update', [chained_updates, chained_updates_for_lp])
@pytest.mark.parametrize('loss', [(10, 1e-6), (1e-6, 10)])
@pytest.mark.parametrize('new_objvals', [(10, 1e-6), (1e-6, 10)])
@pytest.mark.parametrize('C_0_nnz', [.01, 10])
def test_chained_updates(update, loss, new_objvals, C_0_nnz):

    bounds = Bounds(
        objval_min=1,
        objval_max=10,
        loss_min=loss[0],
        loss_max=loss[1],
        L0_min=1,
        L0_max=10
    )

    new_objval_at_feasible, new_objval_at_relaxation = new_objvals

    new_bounds = update(
        bounds, np.tile(C_0_nnz, 100), new_objval_at_feasible, new_objval_at_relaxation
    )

    assert new_bounds.objval_min >= bounds.objval_min
    assert new_bounds.objval_max <= bounds.objval_max

    assert new_bounds.loss_min >= bounds.loss_min
    assert new_bounds.loss_max <= bounds.loss_max

    assert new_bounds.L0_min >= bounds.L0_min
    assert new_bounds.L0_max <= bounds.L0_max
