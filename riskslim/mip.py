from math import ceil, floor
import numpy as np
from cplex import Cplex, SparsePair, infinity as CPX_INFINITY
from .coefficient_set import CoefficientSet
from .helper_functions import get_or_set_default, print_log

#todo: add loss cut
#todo: add constraint function
#todo: default cplex parameters
#todo: check cores
#todo: pass compute_loss to convert_risk_slim_cplex_solution

def create_risk_slim(coef_set, input):
    """
    create RiskSLIM MIP object

    Parameters
    ----------
    input - dictionary of RiskSLIM parameters and formulation

    Returns
    -------
    mip - RiskSLIM surrogate MIP without 0 cuts

    Issues
    ----
    no support for non-integer Lset "values"
    only drops intercept index for variable_names that match '(Intercept)'
    """
    assert isinstance(coef_set, CoefficientSet)
    assert isinstance(input, dict)

    # setup printing and loading
    function_print_flag = input.get('print_flag', False)
    print_from_function = lambda msg: print_log(msg) if function_print_flag else lambda msg: None
    update_parameter = lambda pname, pvalue: get_or_set_default(input, pname, pvalue, print_flag = function_print_flag)

    # set default parameters
    input = update_parameter('w_pos', 1.0)
    input = update_parameter('w_neg', 2.0 - input['w_pos'])
    input = update_parameter('C_0', 0.01)
    input = update_parameter('include_auxillary_variable_for_objval', True)
    input = update_parameter('include_auxillary_variable_for_L0_norm', True)
    input = update_parameter('loss_min', 0.00)
    input = update_parameter('loss_max', float(CPX_INFINITY))
    input = update_parameter('L0_min', 0)
    input = update_parameter('L0_max', len(coef_set))
    input = update_parameter('objval_min', 0.00)
    input = update_parameter('objval_max', float(CPX_INFINITY))
    input = update_parameter('relax_integer_variables', False)
    input = update_parameter('drop_variables', True)
    input = update_parameter('tight_formulation', False)
    input = update_parameter('set_cplex_cutoffs', True)

    # variables
    P = len(coef_set)
    w_pos, w_neg = input['w_pos'], input['w_neg']
    C_0j = np.copy(coef_set.c0)
    L0_reg_ind = np.isnan(C_0j)
    C_0j[L0_reg_ind] = input['C_0']
    C_0j = C_0j.tolist()
    C_0_rho = np.copy(C_0j)
    trivial_L0_min = 0
    trivial_L0_max = np.sum(L0_reg_ind)

    rho_ub = list(coef_set.ub)
    rho_lb = list(coef_set.lb)
    rho_type = ''.join(list(coef_set.vtype))

    # calculate min/max values for loss
    loss_min = max(0.0, float(input['loss_min']))
    loss_max = min(CPX_INFINITY, float(input['loss_max']))

    # calculate min/max values for model size
    L0_min = max(input['L0_min'], 0.0)
    L0_max = min(input['L0_max'], trivial_L0_max)
    L0_min = ceil(L0_min)
    L0_max = floor(L0_max)
    assert L0_min <= L0_max

    # calculate min/max values for objval
    objval_min = max(input['objval_min'], 0.0)
    objval_max = min(input['objval_max'], CPX_INFINITY)
    assert objval_min <= objval_max

    # include constraint on min/max model size?
    nontrivial_L0_min = L0_min > trivial_L0_min
    nontrivial_L0_max = L0_max < trivial_L0_max
    include_auxillary_variable_for_L0_norm = input['include_auxillary_variable_for_L0_norm'] or \
                                             nontrivial_L0_min or \
                                             nontrivial_L0_max

    # include constraint on min/max objective value?
    nontrivial_objval_min = objval_min > 0.0
    nontrivial_objval_max = objval_max < CPX_INFINITY
    include_auxillary_variable_for_objval = input['include_auxillary_variable_for_objval'] or \
                                            nontrivial_objval_min or \
                                            nontrivial_objval_max

    has_intercept = '(Intercept)' in coef_set.variable_names
    """
    RiskSLIM MIP Formulation
    
    minimize w_pos*loss_pos + w_neg *loss_minus + 0*rho_j + C_0j*alpha_j
    
    such that 
    
    L0_min <= L0 <= L0_max
    -rho_min * alpha_j < lambda_j < rho_max * alpha_j

    L_0 in 0 to P
    rho_j in [rho_min_j, rho_max_j]
    alpha_j in {0,1}

    x = [loss_pos, loss_neg, rho_j, alpha_j]

    optional constraints:
    objval = w_pos * loss_pos + w_neg * loss_min + sum(C_0j * alpha_j) (required for callback)
    L0_norm = sum(alpha_j) (required for callback)


    Changes for Tight Formulation (included when input['tight_formulation'] = True):

    sigma_j in {0,1} for j s.t. lambda_j has free sign and alpha_j exists
    lambda_j >= delta_pos_j if alpha_j = 1 and sigma_j = 1
    lambda_j <= -delta_neg_j if alpha_j = 1 and sigma_j = 0
    lambda_j >= alpha_j for j such that lambda_j >= 0
    lambda_j <= -alpha_j for j such that lambda_j <= 0
    
    """

    # create MIP object
    mip = Cplex()
    vars = mip.variables
    cons = mip.linear_constraints

    # set sense
    mip.objective.set_sense(mip.objective.sense.minimize)

    # add main variables
    loss_obj = [w_pos]
    loss_ub = [loss_max]
    loss_lb = [loss_min]
    loss_type = 'C'
    loss_names = ['loss']

    obj = loss_obj + [0.0] * P + C_0j
    ub = loss_ub + rho_ub + [1.0] * P
    lb = loss_lb + rho_lb + [0.0] * P
    ctype = loss_type + rho_type + 'B' * P

    rho_names = ['rho_%d' % j for j in range(P)]
    alpha_names = ['alpha_%d' % j for j in range(P)]
    varnames = loss_names + rho_names + alpha_names

    if include_auxillary_variable_for_objval:
        objval_auxillary_name = ['objval']
        objval_auxillary_ub = [objval_max]
        objval_auxillary_lb = [objval_min]
        objval_type = 'C'

        print_from_function("adding auxiliary variable for objval s.t. %1.4f <= objval <= %1.4f" % (objval_min, objval_max))
        obj += [0.0]
        ub += objval_auxillary_ub
        lb += objval_auxillary_lb
        varnames += objval_auxillary_name
        ctype += objval_type


    if include_auxillary_variable_for_L0_norm:
        L0_norm_auxillary_name = ['L0_norm']
        L0_norm_auxillary_ub = [L0_max]
        L0_norm_auxillary_lb = [L0_min]
        L0_norm_type = 'I'

        print_from_function("adding auxiliary variable for L0_norm s.t. %d <= L0_norm <= %d" % (L0_min, L0_max))
        obj += [0.0]
        ub += L0_norm_auxillary_ub
        lb += L0_norm_auxillary_lb
        varnames += L0_norm_auxillary_name
        ctype += L0_norm_type

    if input['relax_integer_variables']:
        ctype = ctype.replace('I', 'C')
        ctype = ctype.replace('B', 'C')

    vars.add(obj = obj, lb = lb, ub = ub, types = ctype, names = varnames)

    # 0-Norm LB Constraints:
    # lambda_j,lb * alpha_j <= lambda_j <= Inf
    # 0 <= lambda_j - lambda_j,lb * alpha_j < Inf
    for j in range(P):
        cons.add(names = ["L0_norm_lb_" + str(j)],
                 lin_expr = [SparsePair(ind=[rho_names[j], alpha_names[j]], val=[1.0, -rho_lb[j]])],
                 senses = "G",
                 rhs = [0.0])

    # 0-Norm UB Constraints:
    # lambda_j <= lambda_j,ub * alpha_j
    # 0 <= -lambda_j + lambda_j,ub * alpha_j
    for j in range(P):
        cons.add(names = ["L0_norm_ub_" + str(j)],
                 lin_expr =[SparsePair(ind=[rho_names[j], alpha_names[j]], val=[-1.0, rho_ub[j]])],
                 senses = "G",
                 rhs = [0.0])

    # objval_max constraint
    # loss_var + sum(C_0j .* alpha_j) <= objval_max
    if include_auxillary_variable_for_objval:
        print_from_function("adding constraint so that objective value <= " + str(objval_max))
        cons.add(names = ["objval_def"],
                 lin_expr = [SparsePair(ind = objval_auxillary_name + loss_names + alpha_names, val=[-1.0] + loss_obj + C_0j)],
                 senses = "E",
                 rhs = [0.0])

    # Auxiliary L0_norm variable definition:
    # L0_norm = sum(alpha_j)
    # L0_norm - sum(alpha_j) = 0
    if include_auxillary_variable_for_L0_norm:
        cons.add(names = ["L0_norm_def"],
                 lin_expr = [SparsePair(ind = L0_norm_auxillary_name + alpha_names, val = [1.0] + [-1.0] * P)],
                 senses = "E",
                 rhs = [0.0])


    # drop L0_norm_lb constraint for any variable with rho_lb >= 0
    dropped_variables = []
    constraints_to_drop = []

    # drop alpha / L0_norm_ub / L0_norm_lb for ('Intercept')
    if input['drop_variables']:
        # drop L0_norm_ub/lb constraint for any variable with rho_ub/rho_lb >= 0
        sign_pos_ind = np.flatnonzero(coef_set.sign > 0)
        sign_neg_ind = np.flatnonzero(coef_set.sign < 0)
        constraints_to_drop.extend(["L0_norm_lb_" + str(j) for j in sign_pos_ind])
        constraints_to_drop.extend(["L0_norm_ub_" + str(j) for j in sign_neg_ind])

        # drop alpha for any variable where rho_ub = rho_lb = 0
        fixed_value_ind = np.flatnonzero(coef_set.ub == coef_set.lb)
        variables_to_drop = ["alpha_" + str(j) for j in fixed_value_ind]
        vars.delete(variables_to_drop)
        dropped_variables += variables_to_drop
        alpha_names = [alpha_names[j] for j in range(P) if alpha_names[j] not in dropped_variables]

    if has_intercept:
        intercept_idx = coef_set.variable_names.index('(Intercept)')
        intercept_alpha_name = 'alpha_' + str(intercept_idx)
        vars.delete([intercept_alpha_name])

        alpha_names.remove(intercept_alpha_name)
        dropped_variables.append(intercept_alpha_name)

        print_from_function("dropped L0 indicator for '(Intercept)'")
        constraints_to_drop.extend(["L0_norm_ub_" + str(intercept_idx), "L0_norm_lb_" + str(intercept_idx)])

    if len(constraints_to_drop) > 0:
        constraints_to_drop = list(set(constraints_to_drop))
        cons.delete(constraints_to_drop)

    # indices
    indices = {
        'n_variables': vars.get_num(),
        'n_constraints': cons.get_num(),
        'names': vars.get_names(),
        'loss_names': loss_names,
        'rho_names': rho_names,
        'alpha_names': alpha_names,
        'loss': vars.get_indices(loss_names),
        'rho': vars.get_indices(rho_names),
        'alpha': vars.get_indices(alpha_names),
        'L0_reg_ind': L0_reg_ind,
        'C_0_rho': C_0_rho,
        'C_0_alpha': mip.objective.get_linear(alpha_names) if len(alpha_names) > 0 else [],
        }

    if include_auxillary_variable_for_objval:
        indices.update({
            'objval_name': objval_auxillary_name,
            'objval': vars.get_indices(objval_auxillary_name)[0],
            })

    if include_auxillary_variable_for_L0_norm:
        indices.update({
            'L0_norm_name': L0_norm_auxillary_name,
            'L0_norm': vars.get_indices(L0_norm_auxillary_name)[0],
            })

    # officially change the problem to LP if variables are relaxed
    if input['relax_integer_variables']:
        old_problem_type = mip.problem_type[mip.get_problem_type()]
        mip.set_problem_type(mip.problem_type.LP)
        new_problem_type = mip.problem_type[mip.get_problem_type()]
        print_from_function("changed problem type from %s to %s" % (old_problem_type, new_problem_type))

    if input['set_cplex_cutoffs'] and not input['relax_integer_variables']:
        mip.parameters.mip.tolerances.lowercutoff.set(objval_min)
        mip.parameters.mip.tolerances.uppercutoff.set(objval_max)

    return mip, indices


def set_cplex_mip_parameters(cpx, param, display_cplex_progress = False):
    """
    Helper function to set CPLEX parameters of CPLEX MIP object

    Parameters
    ----------
    mip
    param
    display_cplex_progress

    Returns
    -------
    MIP with parameters

    """
    p = cpx.parameters
    p.randomseed.set(param['randomseed'])
    p.threads.set(param['n_cores'])
    p.output.clonelog.set(0)
    p.parallel.set(1)

    if display_cplex_progress is (None or False):
        cpx = set_cpx_display_options(cpx, display_mip = False, display_lp = False, display_parameters = False)

    problem_type = cpx.problem_type[cpx.get_problem_type()]
    if problem_type == 'MIP':
        # CPLEX Memory Parameters
        # MIP.Param.workdir.Cur  = exp_workdir;
        # MIP.Param.workmem.Cur                    = cplex_workingmem;
        # MIP.Param.mip.strategy.file.Cur          = 2; %nodefile uncompressed
        # MIP.Param.mip.limits.treememory.Cur      = cplex_nodefilesize;

        # CPLEX MIP Parameters
        p.emphasis.mip.set(param['mipemphasis'])
        p.mip.tolerances.mipgap.set(param['mipgap'])
        p.mip.tolerances.absmipgap.set(param['absmipgap'])
        p.mip.tolerances.integrality.set(param['integrality_tolerance'])

        # CPLEX Solution Pool Parameters
        p.mip.limits.repairtries.set(param['repairtries'])
        p.mip.pool.capacity.set(param['poolsize'])
        p.mip.pool.replace.set(param['poolreplace'])
        # 0 = replace oldest /1: replace worst objective / #2 = replace least diverse solutions

    return cpx


def set_cpx_display_options(cpx, display_mip = True, display_parameters = False, display_lp = False):

    cpx.parameters.mip.display.set(display_mip)
    cpx.parameters.simplex.display.set(display_lp)

    try:
        cpx.parameters.paramdisplay.set(display_parameters)
    except AttributeError:
        pass

    if not (display_mip or display_lp):
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)

    return cpx


def add_mip_starts(mip, indices, pool, max_mip_starts = float('inf'), mip_start_effort_level = 4):
    """

    Parameters
    ----------
    mip - RiskSLIM surrogate MIP
    indices - indices of RiskSLIM surrogate MIP
    pool - solution pool
    max_mip_starts - max number of mip starts to add (optional; default is add all)
    mip_start_effort_level - effort that CPLEX will spend trying to fix (optional; default is 4)

    Returns
    -------

    """
    # todo remove suboptimal using pool filter
    assert isinstance(mip, Cplex)

    try:
        obj_cutoff = mip.parameters.mip.tolerances.uppercutoff.get()
    except:
        obj_cutoff = float('inf')

    pool = pool.distinct().sort()

    n_added = 0
    for objval, rho in zip(pool.objvals, pool.solutions):
        if np.less_equal(objval, obj_cutoff):
            mip_start_name = "mip_start_" + str(n_added)
            mip_start_obj, _ = convert_to_risk_slim_cplex_solution(rho = rho, indices = indices, objval = objval)
            mip_start_obj = cast_mip_start(mip_start_obj, mip)
            mip.MIP_starts.add(mip_start_obj, mip_start_effort_level, mip_start_name)
            n_added += 1

        if n_added >= max_mip_starts:
            break

    return mip


def cast_mip_start(mip_start, cpx):
    """
    casts the solution values and indices in a Cplex SparsePair

    Parameters
    ----------
    mip_start cplex SparsePair
    cpx Cplex

    Returns
    -------
    Cplex SparsePair where the indices are integers and the values for each variable match the variable type specified in CPLEX Object
    """

    assert isinstance(cpx, Cplex)
    assert isinstance(mip_start, SparsePair)
    vals = list(mip_start.val)
    idx = np.array(list(mip_start.ind), dtype = int).tolist()
    types = cpx.variables.get_types(idx)

    for j, t in enumerate(types):
        if t in ['B', 'I']:
            vals[j] = int(vals[j])
        elif t in ['C']:
            vals[j] = float(vals[j])

    return SparsePair(ind = idx, val = vals)


def convert_to_risk_slim_cplex_solution(rho, indices, loss = None, objval = None):
    """
    Convert coefficient vector 'rho' into a solution for RiskSLIM CPLEX MIP

    Parameters
    ----------
    rho
    indices
    loss
    objval

    Returns
    -------

    """
    global compute_loss
    n_variables = indices['n_variables']
    solution_idx = np.arange(n_variables)
    solution_val = np.zeros(n_variables)

    # rho
    solution_val[indices['rho']] = rho

    # alpha
    alpha = np.zeros(len(indices['alpha']))
    alpha[np.flatnonzero(rho[indices['L0_reg_ind']])] = 1.0
    solution_val[indices['alpha']] = alpha
    L0_penalty = np.sum(indices['C_0_alpha'] * alpha)

    # add loss / objval
    need_loss = 'loss' in indices
    need_objective_val = 'objval' in indices
    need_L0_norm = 'L0_norm' in indices
    need_sigma = 'sigma_names' in indices

    # check that we have the right length
    # COMMENT THIS OUT FOR DEPLOYMENT
    # if need_sigma:
    #     pass
    # else:
    #     assert (indices['n_variables'] == (len(rho) + len(alpha) + need_loss + need_objective_val + need_L0_norm))

    if need_loss:
        if loss is None:
            if objval is None:
                loss = compute_loss(rho)
            else:
                loss = objval - L0_penalty

        solution_val[indices['loss']] = loss

    if need_objective_val:
        if objval is None:
            if loss is None:
                objval = compute_loss(rho) + L0_penalty
            else:
                objval = loss + L0_penalty

        solution_val[indices['objval']] = objval

    if need_L0_norm:
        solution_val[indices['L0_norm']] = np.sum(alpha)

    if need_sigma:
        rho_for_sigma = np.array([indices['rho'][int(s.strip('sigma_'))] for s in indices['sigma_names']])
        solution_val[indices['sigma']] = np.abs(solution_val[rho_for_sigma])

    solution_cpx = SparsePair(ind = solution_idx, val = solution_val.tolist())
    return solution_cpx, objval
