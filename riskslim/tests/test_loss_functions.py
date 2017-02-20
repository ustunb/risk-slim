#noinspection
import numpy as np
import riskslim.loss_functions.log_loss as normal
import riskslim.loss_functions.fast_log_loss as fast
import riskslim.loss_functions.lookup_log_loss as lookup
import riskslim.loss_functions.log_loss_weighted as weighted

#np.__config__.show()

np.random.seed(seed = 0)

#initialize data matrix X and label vector Y
n_rows = 1000000
n_cols = 20
rho_ub = 100
rho_lb = -100


X = np.random.randint(low=0, high=2, size=(n_rows, n_cols))
Y = np.random.randint(low=0, high=2, size=(n_rows, 1))
pos_ind = Y == 1
Y[~pos_ind] = -1
Z = X*Y
Z = np.require(Z, requirements = ['F'], dtype = np.float64)

#setup weights
w_pos = 1.0
w_neg = 1.0
w_total = w_pos + w_neg
w_pos = 2.0 * (w_pos / w_total)
w_neg = 2.0 * (w_neg / w_total)
weights = np.empty_like(Y)
weights[pos_ind] = w_pos
weights[~pos_ind] = w_neg
weights = weights.flatten()


rho = np.random.randint(low=rho_lb, high=rho_ub, size= n_cols)
rho = np.require(rho, dtype=Z.dtype, requirements = ['F'])
set_to_zero = np.random.choice(range(0, n_cols), size = int(np.floor(n_cols/2)), replace=False)
rho[set_to_zero] = 0.0
L0_reg_ind = np.ones(n_cols, dtype = 'bool')
L0_reg_ind[0] = False

#create lookup table
Z_min = np.min(Z, axis = 0)
Z_max = np.max(Z, axis = 0)

def get_score_bounds(Z_min, Z_max, rho):
    pos_ind = np.where(rho>0.0)[0]
    neg_ind = np.where(rho<0.0)[0]
    s_min, s_max = 0, 0

    for j in pos_ind:
        s_max += rho[j] * Z_max[j]
        s_min += rho[j] * Z_min[j]

    for j in neg_ind:
        s_max += rho[j] * Z_min[j]
        s_min += rho[j] * Z_max[j]

    return s_min, s_max


def get_score_bounds_from_range(Z_min, Z_max, rho_lb, rho_ub, L0_max = None):
    "global variables: L0_reg_ind"
    edge_values = np.vstack([Z_min * rho_lb,
                             Z_max * rho_lb,
                             Z_min * rho_ub,
                             Z_max * rho_ub])

    if L0_max is None or L0_max == Z_min.shape[0]:
        s_min = np.sum(np.min(edge_values, axis = 0))
        s_max = np.sum(np.max(edge_values, axis = 0))
    else:
        min_values = np.min(edge_values, axis = 0)
        s_min_reg = np.sum(np.sort(min_values[L0_reg_ind])[0:L0_max])
        s_min_no_reg = np.sum(min_values[~L0_reg_ind])
        s_min = s_min_reg + s_min_no_reg

        max_values = np.max(edge_values, axis = 0)
        s_max_reg = np.sum(-np.sort(-max_values[L0_reg_ind])[0:L0_max])
        s_max_no_reg = np.sum(max_values[~L0_reg_ind])
        s_max = s_max_reg + s_max_no_reg

    return s_min, s_max

min_score, max_score = get_score_bounds_from_range(Z_min, Z_max, rho_lb, rho_ub, L0_max = n_cols)
loss_value_tbl, prob_value_tbl, loss_tbl_offset = lookup.get_loss_value_and_prob_tables(min_score, max_score)
loss_tbl_offset = int(loss_tbl_offset)

#assert correctnes of log_loss from scores function
for s in range(int(min_score), int(max_score)+1):
    normal_value = normal.log_loss_value_from_scores(np.array(s, dtype = Z.dtype, ndmin = 1)) #loss_value_tbl[s+loss_tbl_offset]
    cython_value = fast.log_loss_value_from_scores(np.array(s, dtype = Z.dtype, ndmin = 1))
    table_value = loss_value_tbl[s+loss_tbl_offset]
    lookup_value = lookup.log_loss_value_from_scores(np.array(s,dtype = Z.dtype, ndmin = 1), loss_value_tbl, loss_tbl_offset)
    assert(np.isclose(normal_value, cython_value, rtol = 1e-06))
    assert(np.isclose(table_value, cython_value, rtol = 1e-06))
    assert(np.isclose(table_value, normal_value, rtol = 1e-06))
    assert(np.equal(table_value, lookup_value))

print "all tests passed"

#python implementations need to be 'C' aligned instead of D aligned
Z_py = np.require(Z, requirements = ['C'])
rho_py = np.require(rho, requirements = ['C'])
scores_py = Z_py.dot(rho_py)

#define wrappers
def normal_value_test(): return normal.log_loss_value(Z_py, rho_py)
def fast_value_test(): return fast.log_loss_value(Z, rho)
def lookup_value_test(): return lookup.log_loss_value(Z, rho, loss_value_tbl, loss_tbl_offset)


def normal_cut_test(): return normal.log_loss_value_and_slope(Z_py, rho_py)
def fast_cut_test(): return fast.log_loss_value_and_slope(Z, rho)
def lookup_cut_test(): return lookup.log_loss_value_and_slope(Z, rho, loss_value_tbl, prob_value_tbl, loss_tbl_offset)

# def dynamic_lookup_value_test():
#     s_min_dynamic, s_max_dynamic = get_score_bounds(Z_min, Z_max, rho)
#     tbl, offset = lookup.get_loss_value_table(s_min_dynamic, s_max_dynamic)
#     return lookup.log_loss_value(Z, rho, tbl, offset)

#check values and cuts
normal_cut = normal_cut_test()
cython_cut = fast_cut_test()
lookup_cut = lookup_cut_test()
assert(np.isclose(fast_value_test(), lookup_value_test()))
assert(np.isclose(normal_cut[0], cython_cut[0]))
assert(np.isclose(lookup_cut[0], cython_cut[0]))
assert(all(np.isclose(normal_cut[1], cython_cut[1])))
assert(all(np.isclose(lookup_cut[1], cython_cut[1])))
print "passed cut tests"


#weighted tests
def weighted_value_test(weights): return weighted.log_loss_value(Z_py, weights, np.sum(weights), rho_py)
def weighted_cut_test(weights): return weighted.log_loss_value_and_slope(Z_py, weights, np.sum(weights),  rho_py)
def weighted_scores_test(weights): return weighted.log_loss_value_from_scores(weights, np.sum(weights), scores_py)


#w_pos = w_neg = 1.0
w_pos = 1.0
w_neg = 1.0
w_total = w_pos + w_neg
w_pos = 2.0 * (w_pos / w_total)
w_neg = 2.0 * (w_neg / w_total)
weights = np.empty(Y.shape[0])
pos_ind = Y.flatten() == 1
weights[pos_ind] = w_pos
weights[~pos_ind] = w_neg


weights_match_unit_weights = all(weights == 1.0)

if weights_match_unit_weights:
    print "tests for match between normal and weighted loss function"
    #value
    assert(np.isclose(normal_value_test(), weighted_value_test(weights)))
    assert(np.isclose(normal_value_test(), weighted_scores_test(weights)))

    #cut
    normal_cut = normal_cut_test()
    weighted_cut = weighted_cut_test(weights)
    assert(np.isclose(normal_cut[0], weighted_cut[0]))
    assert(all(np.isclose(normal_cut[1], weighted_cut[1])))

print "passed all tests for weighted implementations when w_pos = w_neg = 1.0"


#w_pos = w_neg = 1.0
w_pos = 0.5 + np.random.rand()
scale_factor = 1 + np.random.rand()
w_neg = 2.0 - w_pos
w_total = w_pos + w_neg
w_pos = 2.0 * (w_pos / w_total)
w_neg = 2.0 * (w_neg / w_total)
pos_ind = Y.flatten() == 1
weights = np.empty(Y.shape[0])
weights[pos_ind] = w_pos
weights[~pos_ind] = w_neg

weighted_value = weighted_value_test(weights)
weighted_cut = weighted_cut_test(weights)
weighted_cut_scaled = weighted_cut_test(scale_factor * weights)
weighted_value_from_scores = weighted_scores_test(weights)

assert(np.isclose(weighted_value, weighted_value_from_scores))
assert(np.isclose(weighted_value, weighted_cut[0]))
assert(np.isclose(weighted_value, weighted_cut_scaled[0]))
assert(np.isclose(weighted_cut[0], weighted_cut_scaled[0]))
assert(np.all(np.isclose(weighted_cut[1], weighted_cut_scaled[1])))

print "passed all tests for weighted loss functions when w_pos = %1.2f and w_neg = %1.2f" % (w_pos, w_neg)





#
# print 'timing for loss value computation \n'
# %timeit -n 20 normal_value = normal_value_test()
# %timeit -n 20 cython_value = fast_value_test()
# %timeit -n 20 lookup_value = lookup_value_test()
#
# print 'timing for loss cut computation \n'
# %timeit -n 20 normal_cut = normal_cut_test()
# %timeit -n 20 cython_cut = fast_cut_test()
# %timeit -n 20 lookup_cut = lookup_cut_test()


