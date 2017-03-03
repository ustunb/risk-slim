import time
import numpy as np
import cplex as cplex
from math import ceil, floor
from cplex import infinity as CPX_INFINITY
from cplex.callbacks import LazyConstraintCallback, HeuristicCallback
from cplex.exceptions import CplexError
import riskslim.loss_functions as lossfun
from .helper_functions import print_log, get_or_set_default, is_integer, cast_to_integer
from .solution_classes import SolutionQueue, SolutionPool
#from .debugging import ipsh #only when debugging

# Lattice CPA
DEFAULT_LCPA_SETTINGS = {
    #
    'c0_value': 1e-6,
    'w_pos': 1.00,
    'tight_formulation': True,  # use a slightly tighter MIP formulation
    #
    # LCPA Settings
    'max_runtime': 300.0,  # max runtime for LCPA
    'max_tolerance': 0.000001,  # tolerance to stop LCPA
    'display_cplex_progress': True,  # setting to True shows CPLEX progress
    'loss_computation': 'normal',  # type of loss computation to use ('normal','fast','lookup')
    'chained_updates_flag': True,  # use chained updates
    'initialization_flag': False,  # use initialization procedure
    'add_cuts_at_heuristic_solutions': True, #add cuts at integer feasible solutions found using polishing/rounding
    # LCPA Rounding Heuristic
    'round_flag': True,  # round continuous solutions with SeqRd
    'polish_rounded_solutions': True,  # polish solutions rounded with SeqRd using DCD
    'rounding_tolerance': float('inf'),  # only solutions with objective value < (1 + tol) are rounded
    'rounding_start_cuts': 0,  # cuts needed to start using rounding heuristic
    'rounding_start_gap': float('inf'),  # optimality gap needed to start using rounding heuristic
    'rounding_stop_cuts': 20000,  # cuts needed to stop using rounding heuristic
    'rounding_stop_gap': 0.2,  # optimality gap needed to stop using rounding heuristic
    #
    # LCPA Polishing Heuristic
    'polish_flag': True,  # polish integer feasible solutions with DCD
    'polishing_tolerance': 0.1, # only solutions with objective value (1 + polishing_ub_to_objval_relgap) are polished. setting to
    'polishing_max_runtime': 10.0,  # max time to run polishing each time
    'polishing_max_solutions': 5.0,  # max # of solutions to polish each time
    'polishing_start_cuts': 0,  # cuts needed to start using polishing heuristic
    'polishing_start_gap': float('inf'),  # min optimality gap needed to start using polishing heuristic
    'polishing_stop_cuts': float('inf'),  # cuts needed to stop using polishing heuristic
    'polishing_stop_gap': 5.0,  # max optimality gap required to stop using polishing heuristic
    #
    # Initialization Procedure
    'init_display_progress': True,  # print progress of initialization procedure
    'init_display_cplex_progress': False,  # print of CPLEX during intialization procedure
    'init_max_runtime': 300.0,  # max time to run CPA in initialization procedure
    'init_max_iterations': 10000,  # max # of cuts needed to stop CPA
    'init_max_tolerance': 0.0001,  # tolerance of solution to stop CPA
    'init_max_runtime_per_iteration': 300.0,  # max time per iteration of CPA
    'init_max_cplex_time_per_iteration': 10.0,  # max time per iteration to solve surrogate problem in CPA
    'init_use_sequential_rounding': True,  # use SeqRd in initialization procedure
    'init_sequential_rounding_max_runtime': 30.0,  # max runtime for SeqRd in initialization procedure
    'init_sequential_rounding_max_solutions': 5,  # max solutions to round using SeqRd
    'init_polishing_after': True,  # polish after rounding
    'init_polishing_max_runtime': 30.0,  # max runtime for polishing
    'init_polishing_max_solutions': 5,  # max solutions to polish
    #
    #  CPLEX Solver Parameters
    'cplex_randomseed': 0,  # random seed
    'cplex_mipemphasis': 0,  # cplex MIP strategy
    'cplex_mipgap': np.finfo('float').eps,  #
    'cplex_absmipgap': np.finfo('float').eps,  #
    'cplex_integrality_tolerance': np.finfo('float').eps,  #
    'cplex_repairtries': 20,  # number of tries to repair user provided solutions
    'cplex_poolsize': 100,  # number of feasible solutions to keep in solution pool
    'cplex_poolrelgap': float('nan'),  # discard if solutions
    'cplex_poolreplace': 2,  # solution pool
    'cplex_n_cores': 1,  # number of cores to use in B & B (must be 1)
    'cplex_nodefilesize': (120 * 1024) / 1,  # node file size
    #
    #  Internal Parameters
    'purge_loss_cuts': False,
    'purge_bound_cuts': False,
}


def run_lattice_cpa(data, constraints, settings=DEFAULT_LCPA_SETTINGS):
    """

    Parameters
    ----------
    data, dict containing training data should pass check_data
    constraints, dict containing 'L0_min, L0_max, CoefficientSet'
    settings

    Returns
    -------
    model_info
    mip_info
    lcpa_info

    """
    global Z, C_0, C_0_nnz, L0_reg_ind, rho_lb, rho_ub
    global compute_loss, compute_loss_cut, compute_loss_from_scores
    global compute_loss_real, compute_loss_cut_real, compute_loss_from_scores_real
    global get_L0_norm, get_L0_penalty, get_alpha, get_L0_penalty_from_alpha, get_L0_range_of_rounded_solution
    global get_objval, is_feasible

    #todo fix initialization procedure
    # initialize settings, replace keys with default values if not found
    settings = dict(settings) if settings is not None else dict()
    settings = {key: settings[key] if key in settings else DEFAULT_LCPA_SETTINGS[key] for key in DEFAULT_LCPA_SETTINGS}

    # separate settings into more manageable objects
    warmstart_settings = {key.lstrip('init_'): settings[key] for key in settings if key.startswith('init_')}
    cplex_parameters = {key.lstrip('cplex_'): settings[key] for key in settings if key.startswith('cplex_')}
    lcpa_settings = {key: settings[key] for key in settings if settings if not (key.startswith('init_') or key.startswith('cplex_'))}

    # data
    N, P = data['X'].shape
    Z = data['X'] * data['Y']

    # sample weights and case weights
    training_weights = data['sample_weights'].flatten()
    training_weights = len(training_weights) * (training_weights / sum(training_weights))
    w_pos = settings['w_pos']
    w_neg = 1.00
    w_total = w_pos + w_neg
    w_pos = 2.00 * (w_pos / w_total)
    w_neg = 2.00 * (w_neg / w_total)
    pos_ind = data['Y'].flatten() == 1
    training_weights[pos_ind] *= w_pos
    training_weights[~pos_ind] *= w_neg

    # trade-off parameter
    c0_value = lcpa_settings['c0_value']
    C_0 = np.array(constraints['coef_set'].C_0j)
    L0_reg_ind = np.isnan(C_0)
    C_0[L0_reg_ind] = c0_value
    C_0_nnz = C_0[L0_reg_ind]

    # constraints
    rho_lb = np.array(constraints['coef_set'].lb)
    rho_ub = np.array(constraints['coef_set'].ub)
    L0_min = constraints['L0_min']
    L0_max = constraints['L0_max']


    # get handles for loss functions
    (compute_loss,
     compute_loss_cut,
     compute_loss_from_scores,
     compute_loss_real,
     compute_loss_cut_real,
     compute_loss_from_scores_real) = load_loss_functions(Z,
                                                          sample_weights = training_weights,
                                                          loss_computation = settings['loss_computation'],
                                                          rho_ub=constraints['coef_set'].ub,
                                                          rho_lb=constraints['coef_set'].lb,
                                                          L0_reg_ind=np.isnan(constraints['coef_set'].C_0j),
                                                          L0_max=constraints['L0_max'])

    # setup function handles for key functions
    def get_L0_norm(rho):
        return np.count_nonzero(rho[L0_reg_ind])

    def is_feasible(rho, lb=rho_lb, ub=rho_ub, L0_min=L0_min, L0_max=L0_max):
        return np.all(ub >= rho) and np.all(lb <= rho) and (L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max)

    def get_L0_penalty(rho):
        return np.sum(C_0_nnz * (rho[L0_reg_ind] != 0.0))

    def get_objval(rho):
        return compute_loss(rho) + np.sum(C_0_nnz * (rho[L0_reg_ind] != 0.0))

    def get_alpha(rho):
        return np.array(abs(rho[L0_reg_ind]) > 0.0, dtype=np.float_)

    def get_L0_penalty_from_alpha(alpha):
        return np.sum(C_0_nnz * alpha)

    def get_L0_range_of_rounded_solution(rho):
        abs_rho = abs(rho)
        rounded_L0_min = get_L0_norm(np.floor(abs_rho))
        rounded_L0_max = get_L0_norm(np.ceil(abs_rho))
        return rounded_L0_min, rounded_L0_max

    # compute bounds on objective value
    bounds = {
        'loss_min': 0.0,
        'loss_max': CPX_INFINITY,
        'objval_min': 0.0,
        'objval_max': CPX_INFINITY,
        'L0_min': constraints['L0_min'],
        'L0_max': constraints['L0_max'],
    }
    bounds['loss_min'], bounds['loss_max'] = get_loss_bounds(Z, rho_ub, rho_lb, L0_reg_ind, L0_max)

    #initialize
    initial_pool = SolutionPool(P)
    initial_cuts = None

    # check if trivial solution is feasible, if so add it to the pool and update bounds
    trivial_solution = np.zeros(P)
    if is_feasible(trivial_solution):
        trivial_objval = compute_loss(trivial_solution)
        bounds['objval_max'] = min(bounds['objval_max'], trivial_objval)
        bounds['loss_max'] = min(bounds['loss_max'], trivial_objval)
        bounds = chained_updates(bounds, C_0_nnz)
        initial_pool = initial_pool.add(objvals=trivial_objval, solutions=trivial_solution)

    # setup risk_slim_lp and risk_slim_mip parameters
    risk_slim_settings = {
        'C_0': c0_value,
        'coef_set': constraints['coef_set'],
        'tight_formulation': lcpa_settings['tight_formulation'],
        'include_auxillary_variable_for_L0_norm': lcpa_settings['chained_updates_flag'],
        'include_auxillary_variable_for_objval': lcpa_settings['chained_updates_flag'],
    }
    risk_slim_settings.update(bounds)

    # run initialization procedure
    if lcpa_settings['initialization_flag']:
        initial_pool, initial_cuts, initial_bounds = initialize_lattice_cpa(risk_slim_settings, warmstart_settings, cplex_parameters, bounds)
        bounds.update(initial_bounds)
        risk_slim_settings.update(initial_bounds)

    # create risk_slim mip
    risk_slim_mip, indices = create_risk_slim(risk_slim_settings)
    risk_slim_mip = set_cplex_mip_parameters(risk_slim_mip, cplex_parameters, display_cplex_progress=lcpa_settings['display_cplex_progress'])
    risk_slim_mip.parameters.timelimit.set(lcpa_settings['max_runtime'])
    indices['C_0_nnz'] = C_0_nnz
    indices['L0_reg_ind'] = L0_reg_ind

    # setup callback functions
    control = {
        'incumbent': np.nan * np.zeros(P),
        'upperbound': float('Inf'),
        'bounds': dict(bounds),
        'lowerbound': 0.0,
        'relative_gap': float('Inf'),
        'nodes_processed': 0,
        'nodes_remaining': 0,
        #
        'start_time': float('nan'),
        'total_run_time': 0.0,
        'total_cut_time': 0.0,
        'total_polish_time': 0.0,
        'total_round_time': 0.0,
        'total_round_then_polish_time': 0.0,
        #
        'cut_callback_times_called': 0,
        'total_cut_callback_time': 0.00,
        'heuristic_callback_times_called': 0,
        'total_heuristic_callback_time': 0.00,
        #
        # number of times solutions were updates
        'n_incumbent_updates': 0,
        'n_heuristic_updates': 0,
        'n_cuts': 0,
        'n_polished': 0,
        'n_rounded': 0,
        'n_rounded_then_polished': 0,
        #
        # total # of bound updates
        'n_update_bounds_calls': 0,
        'n_bound_updates': 0,
        'n_bound_updates_loss_min': 0,
        'n_bound_updates_loss_max': 0,
        'n_bound_updates_L0_min': 0,
        'n_bound_updates_L0_max': 0,
        'n_bound_updates_objval_min': 0,
        'n_bound_updates_objval_max': 0,
    }

    lcpa_cut_queue = SolutionQueue(P)
    lcpa_polish_queue = SolutionQueue(P)
    heuristic_flag = lcpa_settings['round_flag'] or lcpa_settings['polish_flag']
    if heuristic_flag:

        loss_cb = risk_slim_mip.register_callback(LossCallback)
        loss_cb.initialize(indices = indices,
                           control = control,
                           settings = lcpa_settings,
                           initial_cuts=initial_cuts,
                           cut_queue = lcpa_cut_queue,
                           polish_queue = lcpa_polish_queue)

        def heuristic_dcd(rho):
            return(discrete_descent(rho, Z, C_0, rho_ub, rho_lb))

        def heuristic_seqrd(rho, cutoff):
            return(sequential_rounding(rho, Z, C_0, cutoff))

        heuristic_cb = risk_slim_mip.register_callback(PolishAndRoundCallback)
        heuristic_cb.initialize(indices = indices,
                                control = control,
                                settings = lcpa_settings,
                                cut_queue = lcpa_cut_queue,
                                polish_queue = lcpa_polish_queue,
                                dcd_polishing = heuristic_dcd,
                                sequential_rounding = heuristic_seqrd)
    else:
        loss_cb = risk_slim_mip.register_callback(LossCallback)
        loss_cb.initialize(indices = indices,
                           control= control,
                           settings=lcpa_settings,
                           initial_cuts=initial_cuts)

    # attach solution pool
    if len(initial_pool) > 0:
        if lcpa_settings['polish_flag']: # initializing via the polish_queue is preferable since the CPLEX MIP start interface is finicky
            lcpa_polish_queue.add(initial_pool.objvals[0], initial_pool.solutions[0])
        else:
            risk_slim_mip = add_mip_starts(risk_slim_mip, indices, initial_pool, mip_start_effort_level=risk_slim_mip.MIP_starts.effort_level.repair)

        if lcpa_settings['add_cuts_at_heuristic_solutions'] and len(initial_pool) > 1:
            lcpa_cut_queue.add(initial_pool.objvals[1:], initial_pool.solutions[1:])

    # solve using lcpa
    control['start_time'] = time.time()
    risk_slim_mip.solve()
    control['total_run_time'] = time.time() - control['start_time']
    control['cplex_status'] = risk_slim_mip.solution.get_status_string()

    # Results
    incumbent_update_at_termination = False
    try:
        # record the state at the end again since CPLEX may have run for a bit after last callback
        control['found_solution'] = True
        control['upperbound'] = risk_slim_mip.solution.get_objective_value()
        control['lowerbound'] = risk_slim_mip.solution.MIP.get_best_objective()
        try:
            control['relative_gap'] = risk_slim_mip.solution.MIP.get_mip_relative_gap()
        except Exception:
            control['relative_gap'] = (control['upperbound'] - control['lowerbound']) / (
                control['upperbound'] + np.finfo('float').eps)

        incumbent_at_termination = np.array(risk_slim_mip.solution.get_values(indices['rho']))
        incumbent_update_at_termination = not np.array_equal(incumbent_at_termination, control['incumbent'])
        if incumbent_update_at_termination:
            control['incumbent'] = incumbent_at_termination
    except CplexError:
        control['found_solution'] = False

    # get stop reason
    control.pop('start_time')
    control['total_data_time'] = control['total_cut_time'] + control['total_polish_time'] + control['total_round_time'] + control['total_round_then_polish_time']
    control['total_callback_time'] = control['total_cut_callback_time'] + control['total_heuristic_callback_time']
    control['total_solver_time'] = control['total_run_time'] - control['total_callback_time']

    # General Output
    if control['found_solution']:
        model_info = {
            'solution': control['incumbent'],
            'optimality_gap': control['relative_gap'],
            'loss_value': compute_loss(control['incumbent']),
            'objective_value':get_objval(control['incumbent']),
        }
    else:
        model_info = {
            'solution': np.nan * np.zeros(P),
            'optimality_gap': float('inf'),
            'loss_value': float('inf'),
            'objective_value': float('inf')
        }

    model_info.update({
        # lcpa runtime statistics
        'nodes_processed': control['nodes_processed'],
        'run_time': control['total_run_time'],
        'solver_time': control['total_solver_time'],
        'callback_time': control['total_callback_time'],
        'data_time': control['total_data_time'],
        #
        # details about instance
        'c0_value': c0_value,
        'w_pos': w_pos,
    })

    model_info.update(constraints)

    # MIP object
    mip_info = {'risk_slim_mip': risk_slim_mip,
                'risk_slim_idx': indices}

    # LCPA
    lcpa_info = dict(control)
    lcpa_info['bounds'] = dict(bounds)
    lcpa_info['settings'] = dict(settings)

    return model_info, mip_info, lcpa_info


class LossCallback(LazyConstraintCallback):
    """
    This callback has to be initialized after construction using the initialize() method.

    LossCallback is only called when CPLEX finds an integer feasible solution. By default, it will add a cut at this
    solution to improve the cutting-plane approximation of the loss function. The cut is added as a 'lazy' constraint
    into the surrogate LP so that it is evaluated only when necessary.

    Optional functionality:

    - add an initial set of cutting planes found by warm starting
      requires initial_cuts

    - store integer feasible solutions to 'polish' queue so that they can be polished with DCD in the PolishAndRoundCallback
      requires settings['polish_flag'] = True

    - adds cuts at integer feasible solutions found by the PolishAndRoundCallback
      requires settings['add_cuts_at_heuristic_solutions'] = True

    - reduces overall search region by adding constraints on objval_max, l0_max, loss_min, loss_max
      requires settings['chained_updates_flag'] = True
    """

    def initialize(self, indices, control, settings, initial_cuts=None, cut_queue=None, polish_queue=None):

        self.settings = settings  #store pointer to shared settings so that settings can be turned on/off during B&B
        self.control = control  # dict containing information for flow
        self.cut_idx = indices['loss'] + indices['rho']
        self.rho_idx = indices['rho']
        self.alpha_idx = indices['alpha']
        self.L0_reg_ind  = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']

        # todo (validate initial cutting planes)
        self.initial_cuts = initial_cuts

        # cplex has the ability to drop cutting planes that are not used. by default, we force CPLEX to use all cutting planes.
        self.loss_cut_purge_flag = self.use_constraint.purge if self.settings['purge_loss_cuts'] else self.use_constraint.force

        # setup pointer to cut_queue to receive cuts from PolishAndRoundCallback
        if self.settings['add_cuts_at_heuristic_solutions'] and cut_queue is not None:
            self.cut_queue = cut_queue
        else:
            self.cut_queue = SolutionQueue(len(self.rho_idx))

        # setup pointer to polish_queue to send integer solutions to PolishAndRoundCallback
        if self.settings['polish_flag'] and polish_queue is not None:
            self.polish_queue = cut_queue
        else:
            self.polish_queue = SolutionQueue(len(self.rho_idx))

        # for update_bounds
        if self.settings['chained_updates_flag']:
            self.loss_cut_constraint = [indices['loss'], [1.0]]
            self.L0_cut_constraint = [[indices['L0_norm']], [1.0]]
            self.objval_cut_constraint = [[indices['objval']], [1.0]]
            self.bound_cut_purge_flag = self.use_constraint.purge if self.settings['purge_loss_cuts'] else self.use_constraint.force

        return

    def update_bounds(self):

        #print_log('in update bounds')
        bounds = chained_updates(old_bounds = self.control['bounds'],
                                 C_0_nnz = self.C_0_nnz,
                                 new_objval_at_relaxation=self.control['lowerbound'],
                                 new_objval_at_feasible=self.control['upperbound'])

        #add cuts if bounds need to be tighter
        if bounds['loss_min'] > self.control['bounds']['loss_min']:
            self.add(constraint = self.loss_cut_constraint, sense = "G", rhs = bounds['loss_min'], use=self.bound_cut_purge_flag)
            self.control['bounds']['loss_min'] = bounds['loss_min']
            self.control['n_bound_updates_loss_min'] += 1

        if bounds['objval_min'] > self.control['bounds']['objval_min']:
            self.add(constraint = self.objval_cut_constraint, sense = "G", rhs = bounds['objval_min'], use=self.bound_cut_purge_flag)
            self.control['bounds']['objval_min'] = bounds['objval_min']
            self.control['n_bound_updates_objval_min'] += 1

        if bounds['L0_max'] < self.control['bounds']['L0_max']:
            self.add(constraint=self.L0_cut_constraint, sense="L", rhs=bounds['L0_max'], use=self.bound_cut_purge_flag)
            self.control['bounds']['L0_max'] = bounds['L0_max']
            self.control['n_bound_updates_L0_max'] += 1

        if bounds['loss_max'] < self.control['bounds']['loss_max']:
            self.add(constraint=self.loss_cut_constraint, sense="L", rhs=bounds['loss_max'], use=self.bound_cut_purge_flag)
            self.control['bounds']['loss_max'] = bounds['loss_max']
            self.control['n_bound_updates_loss_max'] += 1

        if bounds['objval_max'] < self.control['bounds']['objval_max']:
            self.add(constraint=self.objval_cut_constraint, sense="L", rhs=bounds['objval_max'], use=self.bound_cut_purge_flag)
            self.control['bounds']['objval_max'] = bounds['objval_max']
            self.control['n_bound_updates_objval_max'] += 1

        #print_log('left update bounds')
        return

    def __call__(self):

        #print_log('in cut callback')
        callback_start_time = time.time()

        #record entry metrics
        self.control['cut_callback_times_called'] += 1
        self.control['lowerbound'] = self.get_best_objective_value()
        self.control['relative_gap'] = self.get_MIP_relative_gap()
        self.control['nodes_processed'] = self.get_num_nodes()
        self.control['nodes_remaining'] = self.get_num_remaining_nodes()

        # add initial cuts first time the callback is used
        if self.initial_cuts is not None:
            print_log('adding initial cuts')
            for i in range(len(self.initial_cuts['coefs'])):
                self.add(self.initial_cuts['coefs'][i], "G", rhs=self.initial_cuts['lhs'][i], use=self.loss_cut_purge_flag)
            self.initial_cuts = None

        # get integer feasible solution
        rho = np.array(self.get_values(self.rho_idx))
        alpha = np.array(self.get_values(self.alpha_idx))

        # check that CPLEX is actually integer. if not, then recast as int
        if ~is_integer(rho):
            rho = cast_to_integer(rho)
            alpha = get_alpha(rho)

        # add cutting plane at integer feasible solution
        cut_start_time = time.time()
        loss_value, loss_slope = compute_loss_cut(rho)
        cut_lhs = float(loss_value - loss_slope.dot(rho))
        cut_constraint = [self.cut_idx, [1.0] + (-loss_slope).tolist()]
        self.add(constraint=cut_constraint, sense="G", rhs=cut_lhs, use=self.loss_cut_purge_flag)
        cut_time = time.time() - cut_start_time
        cuts_added = 1

        # if solution updates incumbent, then add solution to queue for polishing
        current_upperbound = float(loss_value + get_L0_penalty_from_alpha(alpha))
        incumbent_update = current_upperbound < self.control['upperbound']

        if incumbent_update:
            self.control['incumbent'] = rho
            self.control['upperbound'] = current_upperbound
            self.control['n_incumbent_updates'] += 1


        if self.settings['polish_flag']:
            polishing_cutoff = self.control['upperbound'] * (1.0 + self.settings['polishing_tolerance'])
            if current_upperbound < polishing_cutoff:
                self.polish_queue.add(current_upperbound, rho)

        # add cutting planes at other integer feasible solutions in cut_queue
        if self.settings['add_cuts_at_heuristic_solutions']:
            if len(self.cut_queue) > 0:
                self.cut_queue.filter_sort_unique()
                for cut_rho in self.cut_queue.solutions:
                    cut_start_time = time.time()
                    [loss_value, loss_slope] = compute_loss_cut(cut_rho)
                    cut_lhs = float(loss_value - loss_slope.dot(cut_rho))
                    cut_constraint = [self.cut_idx, [1.0] + (-loss_slope).tolist()]
                    self.add(constraint=cut_constraint, sense="G", rhs=cut_lhs, use=self.loss_cut_purge_flag)
                    cuts_added += 1
                    cut_time += time.time() - cut_start_time
                self.cut_queue.clear()

        # update bounds
        if self.settings['chained_updates_flag']:
            if (self.control['lowerbound'] > self.control['bounds']['objval_min']) or (self.control['upperbound'] < self.control['bounds']['objval_max']):
                self.control['n_update_bounds_calls'] += 1
                self.update_bounds()

        # record metrics at end
        #print_log('last metrics')
        self.control['n_cuts'] += cuts_added
        self.control['total_cut_time'] += cut_time
        self.control['total_cut_callback_time'] += time.time() - callback_start_time
        #print_log('left cut callback')
        return


class PolishAndRoundCallback(HeuristicCallback):
    """
    This callback has to be initialized after construction using the initialize() method.

    HeuristicCallback is called intermittently during B&B by CPLEX. It runs several heuristics in a fast way and contains
    several options to stop early. Note: It is important for the callback to run quickly since it is called fairly often.
    If HeuristicCallback runs slowly, then it will slow down overall B&B progress.

    Heuristics include:

    - Runs sequential rounding on the continuous solution from the surrogate LP (only if there has been a change in the
      lower bound). Requires settings['round_flag'] = True. If settings['polish_after_rounding'] = True, then the
      rounded solutions are polished using DCD.

    - Polishes integer solutions in polish_queue using DCD. Requires settings['polish_flag'] = True.

    Optional:

    - Feasible solutions are passed to LazyCutConstraintCallback using the cut_queue

    Known issues:

    - Sometimes CPLEX does not return an integer feasible solution (in which case we correct this manually)
    """

    def initialize(self, indices, control, settings, cut_queue, polish_queue, dcd_polishing, sequential_rounding):

        self.rho_idx = indices['rho']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']
        self.indices = indices
        self.previous_lowerbound = 0.0
        self.control = control
        self.settings = settings
        self.cut_queue = cut_queue #pointer to cut_queue

        self.round_flag = settings['round_flag']
        self.polish_rounded_solutions = settings['polish_rounded_solutions']
        self.polish_flag = settings['polish_flag']
        self.polish_queue = polish_queue

        self.dcd_polishing = dcd_polishing
        self.sequential_rounding = sequential_rounding
        return


    def update_heuristic_flags(self):

        # keep on rounding?
        keep_rounding = (self.settings['rounding_start_cuts'] <= self.control['n_cuts'] <= self.settings['rounding_stop_cuts'] and
                         self.settings['rounding_stop_gap'] <= self.control['relative_gap'] <= self.settings['rounding_start_gap'])

        # keep on polishing?
        keep_polishing = (self.settings['polishing_start_cuts'] <= self.control['n_cuts'] <= self.settings['polishing_stop_cuts'] and
                          self.settings['polishing_stop_gap'] <= self.control['relative_gap'] <= self.settings['polishing_start_gap'])

        self.polish_flag = keep_polishing & self.polish_flag
        self.round_flag = keep_rounding & self.round_flag
        self.polish_rounded_solutions = self.round_flag & self.polish_rounded_solutions
        return


    def __call__(self):
        #print_log('in heuristic callback')

        #fast exit if rounding is off & (polishing is off / no solution to polish)
        if (self.round_flag is False) and (self.polish_flag is False or len(self.polish_queue) == 0):
            return

        callback_start_time = time.time()
        self.control['upperbound'] = self.get_incumbent_objective_value()
        self.control['lowerbound'] = self.get_best_objective_value()
        self.control['relative_gap'] = self.get_MIP_relative_gap()

        # check if lower bound was updated since last call
        lowerbound_update = self.previous_lowerbound < self.control['lowerbound']
        if lowerbound_update:
            self.previous_lowerbound = self.control['lowerbound']

        # check if incumbent solution has been updated
        # if incumbent solution is not integer, then recast as integer and update objective value manually
        if self.has_incumbent():
            cplex_incumbent = np.array(self.get_incumbent_values(self.rho_idx))

            cplex_rounding_issue = ~is_integer(cplex_incumbent)
            if cplex_rounding_issue:
                cplex_incumbent = cast_to_integer(cplex_incumbent)

            incumbent_update = not np.array_equal(cplex_incumbent, self.control['incumbent'])

            if incumbent_update:
                self.control['incumbent'] = cplex_incumbent
                self.control['n_incumbent_updates'] += 1
                if cplex_rounding_issue:
                    self.control['upperbound'] = get_objval(cplex_incumbent)

        #decide whether to keep on rounding/polishing
        self.update_heuristic_flags()
        #variables to store best objective value / solution from heuristics
        best_objval = float('inf')
        best_solution = None

        # run sequential rounding if lower bound was updated since the last call
        if self.round_flag and lowerbound_update:

            rho_cts = np.array(self.get_values(self.rho_idx))
            min_l0_norm, max_l0_norm = get_L0_range_of_rounded_solution(rho_cts)
            rounded_solution_is_feasible = (min_l0_norm < self.control['bounds']['L0_max'] and
                                            max_l0_norm > self.control['bounds']['L0_min'])

            if rounded_solution_is_feasible:
                rounding_cutoff = self.control['upperbound'] * (1.0 + self.settings['rounding_tolerance'])
                rounding_start_time = time.time()
                rounded_solution, rounded_objval, early_stop = self.sequential_rounding(rho_cts, rounding_cutoff)
                self.control['n_rounded'] += 1
                self.control['total_round_time'] += time.time() - rounding_start_time

                #round solution is sequential rounding did not stop early
                if not early_stop:

                    if self.settings['add_cuts_at_heuristic_solutions']:
                        self.cut_queue.add(rounded_objval, rounded_solution)

                    if is_feasible(rounded_solution, L0_min=self.control['bounds']['L0_min'], L0_max=self.control['bounds']['L0_max']):
                        best_solution = rounded_solution
                        best_objval = rounded_objval

                    if self.polish_rounded_solutions:
                        current_upperbound = min(rounded_objval, self.control['upperbound'])
                        polishing_cutoff = current_upperbound * (1.0 + self.settings['polishing_tolerance'])

                        if rounded_objval < polishing_cutoff:
                            start_time = time.time()
                            polished_solution, _, polished_objval = self.dcd_polishing(rounded_solution)
                            self.control['n_rounded_then_polished'] += 1
                            self.control['total_round_then_polish_time'] += time.time() - start_time

                            if self.settings['add_cuts_at_heuristic_solutions']:
                                self.cut_queue.add(polished_objval, polished_solution)

                            if is_feasible(polished_solution,
                                           L0_min=self.control['bounds']['L0_min'],
                                           L0_max=self.control['bounds']['L0_max']):
                                best_solution = polished_solution
                                best_objval = polished_objval

        # polish solutions in polish_queue or that were produced by rounding
        if self.polish_flag and len(self.polish_queue) > 0:

            #get current upperbound
            current_upperbound = min(best_objval, self.control['upperbound'])
            polishing_cutoff = current_upperbound * (1.0 + self.settings['polishing_tolerance'])

            #only polish solutions in queue that <= cutoff
            self.polish_queue.filter_sort_unique(polishing_cutoff)
            n_to_polish = min(len(self.polish_queue), self.settings['polishing_max_solutions'])

            if n_to_polish > 0:
                print('moo')
                polish_time = 0
                n_polished = 0
                polished_queue = SolutionQueue(self.polish_queue.P)

                for i in range(n_to_polish):

                    if self.polish_queue.objvals[i] >= polishing_cutoff:
                        break

                    polish_start_time = time.time()
                    polished_solution, _, polished_objval = self.dcd_polishing(self.polish_queue.solutions[i])
                    polish_time += time.time() - polish_start_time
                    n_polished += 1

                    #update cutoff when polishing returns feasible solution
                    if is_feasible(polished_solution, L0_min=self.control['bounds']['L0_min'], L0_max=self.control['bounds']['L0_max']):
                        polished_queue.add(polished_objval, polished_solution)
                        current_upperbound = min(polished_objval, polished_objval)
                        polishing_cutoff = current_upperbound * (1.0 + self.settings['polishing_tolerance'])

                    if polish_time > self.settings['polishing_max_runtime']:
                        break

                if self.settings['add_cuts_at_heuristic_solutions']:
                    self.cut_queue.add(polished_queue.objvals, polished_queue.solutions)

                # check if the best polished solution will improve the queue
                if len(polished_queue) > 0 and np.min(polished_queue.objvals) < best_objval:
                    best_objval, best_solution = polished_queue.get_best_objval_and_solution()

                self.polish_queue.clear()
                self.control['total_polish_time'] += polish_time
                self.control['n_polished'] += n_polished

        # if heuristics produces a better solution then update the incumbent
        heuristic_update = best_objval < self.control['upperbound']
        if heuristic_update:
            # note that solution is not always accepted by CPLEX
            self.control['n_heuristic_updates'] += 1
            proposed_solution, proposed_objval = convert_to_risk_slim_cplex_solution(rho=best_solution, indices=self.indices, objval=best_objval)
            self.set_solution(solution = proposed_solution, objective_value = proposed_objval)

        self.control['total_heuristic_callback_time'] += time.time() - callback_start_time
        #print_log('left heuristic callback')
        return


# CPLEX
def create_risk_slim(input):
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

    assert 'coef_set' in input, 'input is missing coef_set'
    P = len(input['coef_set'])

    # setup printing and loading
    function_print_flag = input['print_flag'] if 'print_flag' in input else False
    print_from_function = lambda msg: print_log(msg) if function_print_flag else lambda msg: None
    update_parameter = lambda pname, pvalue: get_or_set_default(input, pname, pvalue, print_flag=function_print_flag)

    # set default parameters
    input = update_parameter('w_pos', 1.0)
    input = update_parameter('w_neg', 2.0 - input['w_pos'])
    input = update_parameter('C_0', 0.01)
    input = update_parameter('loss_min', 0.00)
    input = update_parameter('loss_max', float(CPX_INFINITY))
    input = update_parameter('include_auxillary_variable_for_L0_norm', False)
    input = update_parameter('L0_min', 0)
    input = update_parameter('L0_max', P)
    input = update_parameter('include_auxillary_variable_for_objval', False)
    input = update_parameter('objval_min', 0.00)
    input = update_parameter('objval_max', float(CPX_INFINITY))
    input = update_parameter('class_based', False)
    input = update_parameter('relax_integer_variables', False)
    input = update_parameter('tight_formulation', False)
    input = update_parameter('set_cplex_cutoffs', True)

    w_pos, w_neg = input['w_pos'], input['w_neg']
    C_0j = np.copy(input['coef_set'].C_0j)
    L0_reg_ind = np.isnan(C_0j)
    C_0j[L0_reg_ind] = input['C_0']
    C_0j = C_0j.tolist()
    C_0_rho = np.copy(C_0j)
    trivial_L0_min = 0
    trivial_L0_max = np.sum(L0_reg_ind)

    rho_lb = input['coef_set'].get_field_as_list('lb')
    rho_ub = input['coef_set'].get_field_as_list('ub')
    rho_type = ''.join(input['coef_set'].get_field_as_list('vtype'))

    # calculate min/max values for loss
    loss_min = max(0.0, float(input['loss_min']))
    loss_max = min(CPX_INFINITY, float(input['loss_max']))

    # calculate min/max values for model size
    L0_min = max(input['L0_min'], 0.0)
    L0_min = ceil(L0_min)
    L0_max = min(input['L0_max'], trivial_L0_max)
    L0_max = floor(L0_max)
    if L0_min > L0_max:
        print_from_function("warning: L0_min > L0_max, setting both to trivial values")
        L0_min = trivial_L0_min
        L0_max = trivial_L0_max

    # calculate min/max values for objval
    objval_min = max(input['objval_min'], 0.0)
    objval_max = min(input['objval_max'], CPX_INFINITY)
    if objval_min > objval_max:
        print_from_function("objval_min > objval_max, setting both to trivial values")
        objval_min = 0.0
        objval_max = CPX_INFINITY

    # include constraint on min/max model size?
    nontrivial_L0_min = L0_min > 0
    nontrivial_L0_max = L0_max < trivial_L0_max
    include_auxillary_variable_for_L0_norm = input[
                                                 'include_auxillary_variable_for_L0_norm'] or nontrivial_L0_min or nontrivial_L0_max

    # include constraint on min/max objective value?
    nontrivial_objval_min = (objval_min > 0.0)
    nontrivial_objval_max = (objval_max < CPX_INFINITY)
    include_auxillary_variable_for_objval = input[
                                                'include_auxillary_variable_for_objval'] or nontrivial_objval_min or nontrivial_objval_max

    # tight formulation flag
    tight_formulation = input['tight_formulation']

    # %min w_pos*loss_pos + w_neg *loss_minus + 0*rho_j + C_0j*alpha_j
    # %s.t.
    # %L0_min <= L0 <= L0_max
    # %-rho_min * alpha_j < lambda_j < rho_max * alpha_j
    #
    # %L_0 in 0 to P
    # %rho_j
    # %lambda_j in Z
    # %alpha_j in B

    # x = [loss_pos, loss_neg, rho_j, alpha_j]

    # optional constraints:
    # objval = w_pos * loss_pos + w_neg * loss_min + sum(C_0j * alpha_j) (required for callback)
    # L0_norm = sum(alpha_j) (required for callback)

    # if tight formulation, add
    # sigma_j in B for j s.t. lambda_j has free sign and alpha_j exists
    # lambda_j >= delta_pos_j if alpha_j = 1 and sigma_j = 1
    # lambda_j <= -delta_neg_j if alpha_j = 1 and sigma_j = 0
    # lambda_j >= alpha_j for j such that lambda_j >= 0
    # lambda_j <= -alpha_j for j such that lambda_j <= 0

    mip = cplex.Cplex()
    mip.objective.set_sense(mip.objective.sense.minimize)

    loss_obj = [w_pos]
    loss_ub = [loss_max]
    loss_lb = [loss_min]
    loss_type = 'C'
    loss_names = ['loss']

    obj = loss_obj + [0.0] * P + C_0j
    ub = loss_ub + rho_ub + [1.0] * P
    lb = loss_lb + rho_lb + [0.0] * P
    ctype = loss_type + rho_type + 'B' * P

    rho_names = ['rho_' + str(j) for j in range(0, P)]
    alpha_names = ['alpha_' + str(j) for j in range(0, P)]
    varnames = loss_names + rho_names + alpha_names

    if include_auxillary_variable_for_objval:
        objval_auxillary_name = ['objval']
        objval_auxillary_ub = [objval_max]
        objval_auxillary_lb = [objval_min]
        objval_type = 'C'
        print_from_function(
            "including auxiliary variable for objval such that " + str(objval_min) + " <= objval <= " + str(objval_max))
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
        print_from_function(
            "including auxiliary variable for L0_norm such that " + str(L0_min) + " <= L0_norm <= " + str(L0_max))
        obj += [0.0]
        ub += L0_norm_auxillary_ub
        lb += L0_norm_auxillary_lb
        varnames += L0_norm_auxillary_name
        ctype += L0_norm_type

    if input['relax_integer_variables']:
        ctype = ctype.replace('I', 'C')
        ctype = ctype.replace('B', 'C')

    mip.variables.add(obj=obj, lb=lb, ub=ub, types=ctype, names=varnames)

    # 0-Norm LB Constraints:
    # lambda_j,lb * alpha_j <= lambda_j <= Inf
    # 0 <= lambda_j - lambda_j,lb * alpha_j < Inf
    for j in range(0, P):
        constraint_name = ["L0_norm_lb_" + str(j)]
        constraint_expr = [cplex.SparsePair(ind=[rho_names[j], alpha_names[j]], val=[1.0, -rho_lb[j]])]
        constraint_rhs = [0.0]
        constraint_sense = "G"
        mip.linear_constraints.add(lin_expr=constraint_expr,
                                   senses=constraint_sense,
                                   rhs=constraint_rhs,
                                   names=constraint_name)

    # 0-Norm UB Constraints:
    # lambda_j <= lambda_j,ub * alpha_j
    # 0 <= -lambda_j + lambda_j,ub * alpha_j
    for j in range(0, P):
        constraint_name = ["L0_norm_ub_" + str(j)]
        constraint_expr = [cplex.SparsePair(ind=[rho_names[j], alpha_names[j]], val=[-1.0, rho_ub[j]])]
        constraint_rhs = [0.0]
        constraint_sense = "G"
        mip.linear_constraints.add(lin_expr=constraint_expr,
                                   senses=constraint_sense,
                                   rhs=constraint_rhs,
                                   names=constraint_name)

    # objval_max constraint
    # loss_var + sum(C_0j .* alpha_j) <= objval_max
    if include_auxillary_variable_for_objval:
        print_from_function("adding constraint so that objective value <= " + str(objval_max))
        constraint_name = ["objval_def"]
        constraint_expr = [cplex.SparsePair(ind=objval_auxillary_name + loss_names + alpha_names,
                                            val=[-1.0] + loss_obj + C_0j)]
        constraint_rhs = [0.0]
        constraint_sense = "E"
        mip.linear_constraints.add(lin_expr=constraint_expr,
                                   senses=constraint_sense,
                                   rhs=constraint_rhs,
                                   names=constraint_name)

    # Auxiliary L0_norm variable definition:
    # L0_norm = sum(alpha_j)
    # L0_norm - sum(alpha_j) = 0
    if include_auxillary_variable_for_L0_norm:
        constraint_name = ["L0_norm_def"]
        constraint_expr = [cplex.SparsePair(ind=L0_norm_auxillary_name + alpha_names,
                                            val=[1.0] + [-1.0] * P)]
        constraint_rhs = [0.0]
        constraint_sense = "E"
        mip.linear_constraints.add(lin_expr=constraint_expr,
                                   senses=constraint_sense,
                                   rhs=constraint_rhs,
                                   names=constraint_name)

    # Tighter Formulation
    if tight_formulation:
        assert (all(input['coef_set'].vtype == 'I'))
        sigma_obj = [0.0] * P
        sigma_ub = [1.0] * P
        sigma_lb = [0.0] * P
        sigma_type = 'B' * P
        sigma_names = ['sigma_' + str(j) for j in range(0, P)]
        mip.variables.add(obj=sigma_obj,
                          ub=sigma_ub,
                          lb=sigma_lb,
                          types=sigma_type,
                          names=sigma_names)
        delta_pos = np.ones(P)
        delta_neg = np.ones(P)
        M_pos = (rho_lb - delta_pos).tolist()
        M_neg = (rho_ub + delta_neg).tolist()

        # add constraint to force lambda[j] >= delta_pos[j] when alpha[j] = 1 and sigma[j] = 1
        # lambda[j] >= delta_pos[j]*alpha[j] + M_pos[j](1-sigma[j])
        # lambda[j] - delta_pos[j]*alpha[j] - M_pos[j] * sigma[j] >= M_pos[j]
        for j in range(0, P):
            constraint_name = ["min_pos_value_when_L0_on_" + str(j)]
            constraint_expr = [cplex.SparsePair(ind=[rho_names[j], alpha_names[j], sigma_names[j]],
                                                val=[1.0, -delta_pos[j], -M_pos[j]])]
            constraint_rhs = [M_pos[j]]
            constraint_sense = "G"
            mip.linear_constraints.add(lin_expr=constraint_expr,
                                       senses=constraint_sense,
                                       rhs=constraint_rhs,
                                       names=constraint_name)

        # add constraint to force lambda[j] <= delta_neg[j] when alpha[j] = 1 and sigma[j] = 0
        # lambda[j] <= -delta_neg[j]*alpha[j] + M_neg[j]*sigma[j]
        # 0 <= -lambda[j] - delta_neg[j]*alpha[j] + M_neg[j]*sigma[j]
        for j in range(0, P):
            constraint_name = ["min_neg_value_when_L0_on_" + str(j)]
            constraint_expr = [cplex.SparsePair(ind=[rho_names[j], alpha_names[j], sigma_names[j]],
                                                val=[-1.0, -delta_neg[j], M_neg[j]])]
            constraint_rhs = [0.0]
            constraint_sense = "G"
            mip.linear_constraints.add(lin_expr=constraint_expr,
                                       senses=constraint_sense,
                                       rhs=constraint_rhs,
                                       names=constraint_name)

    dropped_variables = []
    sign_pos_ind = np.where(input['coef_set'].sign > 0)[0].tolist()
    sign_neg_ind = np.where(input['coef_set'].sign < 0)[0].tolist()
    fixed_value_ind = np.where(input['coef_set'].ub == input['coef_set'].lb)[0].tolist()

    # drop L0_norm_lb constraint for any variable with rho_lb >= 0
    constraints_to_drop = ["L0_norm_lb_" + str(j) for j in sign_pos_ind]
    mip.linear_constraints.delete(constraints_to_drop)

    # drop L0_norm_ub constraint for any variable with rho_ub >= 0
    constraints_to_drop = ["L0_norm_ub_" + str(j) for j in sign_neg_ind]
    mip.linear_constraints.delete(constraints_to_drop)

    # drop alpha for any variable where rho_ub = rho_lb = 0
    variables_to_drop = ["alpha_" + str(j) for j in fixed_value_ind]
    mip.variables.delete(variables_to_drop)
    dropped_variables += variables_to_drop
    alpha_names = [alpha_names[j] for j in range(0, P) if alpha_names[j] not in dropped_variables]

    # drop alpha, L0_norm_ub and L0_norm_lb for ('Intercept')
    try:
        offset_idx = input['coef_set'].get_field_as_list('variable_names').index('(Intercept)')
        variables_to_drop = ['alpha_' + str(offset_idx)]
        mip.variables.delete(variables_to_drop)
        alpha_names.pop(alpha_names.index('alpha_' + str(offset_idx)))
        dropped_variables += ['alpha_' + str(offset_idx)]
        print_from_function("dropped L0 variable for intercept variable")
    except CplexError:
        pass

    try:
        offset_idx = input['coef_set'].get_field_as_list('variable_names').index('(Intercept)')
        mip.linear_constraints.delete(["L0_norm_lb_" + str(offset_idx),
                                       "L0_norm_ub_" + str(offset_idx)])
        print_from_function("dropped L0 constraints for intercept variable")
    except CplexError:
        pass

    # drop variables that were added for a tighter formulation
    if tight_formulation:
        dropped_alphas = [j for j in range(0, P) if ("alpha_" + str(j)) in dropped_variables]

        # drop min_neg_value_when_L0_on_j for any variable with rho_lb >= 0 or that does not have alpha[j]
        drop_ind = list(set(sign_pos_ind + dropped_alphas))
        constraints_to_drop = ["min_neg_value_when_L0_on_" + str(j) for j in drop_ind]
        mip.linear_constraints.delete(constraints_to_drop)

        # drop min_pos_value_when_L0_on_j constraint for any variable with rho_ub >= 0 or that does not have alpha[j]
        drop_ind = list(set(sign_neg_ind + dropped_alphas))
        constraints_to_drop = ["min_pos_value_when_L0_on_" + str(j) for j in drop_ind]
        mip.linear_constraints.delete(constraints_to_drop)

        # drop sigma_j for any variable with fixed sign or that does not have alpha[j]
        sigma_to_drop_ind = list(set(sign_pos_ind + sign_neg_ind + dropped_alphas))
        variables_to_drop = ["sigma_" + str(j) for j in sigma_to_drop_ind]
        mip.variables.delete(variables_to_drop)
        dropped_variables += variables_to_drop
        sigma_names = [sigma_names[j] for j in range(0, P) if sigma_names[j] not in dropped_variables]

        # for variables with sign[j] > 0 where alpha exists, add constraint again
        # lambda[j] >= delta_pos * alpha[j]
        # lambda[j] - delta_pos[j] * alpha[j] >= 0
        add_pos_con_ind_ = [j for j in range(0, P) if
                            j in sign_pos_ind and
                            ("sigma_" + str(j)) in dropped_variables
                            ("alpha_" + str(j)) not in dropped_variables]

        for j in add_pos_con_ind_:
            constraint_name = ["min_pos_value_when_L0_on_" + str(j)]
            constraint_expr = [cplex.SparsePair(ind=[rho_names[j], alpha_names[j]],
                                                val=[1.0, -delta_pos[j]])]
            constraint_rhs = [0.0]
            constraint_sense = "G"
            mip.linear_constraints.add(lin_expr=constraint_expr,
                                       senses=constraint_sense,
                                       rhs=constraint_rhs,
                                       names=constraint_name)

        # for variables with sign[j] < 0 where alpha exists, add constraint again
        # lambda[j] <= -delta_neg[j] * alpha[j]
        # -lambda[j] - delta_neg[j] * alpha[j] >= 0
        add_neg_con_ind_ = [j for j in range(0, P) if
                            j in sign_neg_ind and
                            ("sigma_" + str(j)) in dropped_variables
                            ("alpha_" + str(j)) not in dropped_variables]

        for j in add_neg_con_ind_:
            constraint_name = ["min_neg_value_when_L0_on_" + str(j)]
            constraint_expr = [cplex.SparsePair(ind=[rho_names[j], alpha_names[j]],
                                                val=[-1.0, -delta_neg[j]])]
            constraint_rhs = [0.0]
            constraint_sense = "G"
            mip.linear_constraints.add(lin_expr=constraint_expr,
                                       senses=constraint_sense,
                                       rhs=constraint_rhs,
                                       names=constraint_name)

    rho_idx = mip.variables.get_indices(rho_names)
    alpha_idx = mip.variables.get_indices(alpha_names)
    if len(alpha_idx) == 0:
        C_0_alpha = []
    else:
        C_0_alpha = mip.objective.get_linear(alpha_idx)

    # indices
    indices = {
        "names": mip.variables.get_names(),
        "rho": rho_idx,
        "rho_names": rho_names,
        "alpha": alpha_idx,
        "alpha_names": alpha_names,
        "L0_reg_ind": L0_reg_ind,
        "C_0_rho": C_0_rho,
        "C_0_alpha": np.array(C_0_alpha),
        "n_variables": mip.variables.get_num(),
        "n_constraints": mip.linear_constraints.get_num(),
    }

    indices["loss"] = [mip.variables.get_indices("loss")]
    indices["loss_names"] = loss_names

    if include_auxillary_variable_for_objval:
        indices['objval'] = mip.variables.get_indices("objval")
        indices['objval_name'] = objval_auxillary_name

    if include_auxillary_variable_for_L0_norm:
        indices['L0_norm'] = mip.variables.get_indices("L0_norm")
        indices['L0_norm_name'] = L0_norm_auxillary_name

    if tight_formulation:
        indices["sigma"] = mip.variables.get_indices(sigma_names)
        indices["sigma_names"] = sigma_names

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


def convert_to_risk_slim_cplex_solution(rho, indices, loss=None, objval=None):
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
    solution_idx = range(0, indices['n_variables'])
    solution_val = np.zeros(indices['n_variables'])

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

    solution_cpx = cplex.SparsePair(ind=solution_idx, val=solution_val.tolist())
    return solution_cpx, objval


def set_cplex_mip_parameters(mip, cplex_parameters, display_cplex_progress=False):
    """
    Helper function to set CPLEX parameters of CPLEX MIP object

    Parameters
    ----------
    mip
    cplex_parameters
    display_cplex_progress

    Returns
    -------
    MIP with parameters

    """
    problem_type = mip.problem_type[mip.get_problem_type()]
    mip.parameters.randomseed.set(cplex_parameters['randomseed'])
    mip.parameters.threads.set(cplex_parameters['n_cores'])
    mip.parameters.output.clonelog.set(0)
    mip.parameters.parallel.set(1)

    if display_cplex_progress is (None or False):
        mip.set_results_stream(None)
        mip.set_log_stream(None)

    if problem_type is 'MIP':

        if display_cplex_progress is (None or False):
            mip.parameters.mip.display.set(0)

        # CPLEX Memory Parameters
        # MIP.Param.workdir.Cur  = exp_workdir;
        # MIP.Param.workmem.Cur                    = cplex_workingmem;
        # MIP.Param.mip.strategy.file.Cur          = 2; %nodefile uncompressed
        # MIP.Param.mip.limits.treememory.Cur      = cplex_nodefilesize;

        # CPLEX MIP Parameters
        mip.parameters.emphasis.mip.set(cplex_parameters['mipemphasis'])
        mip.parameters.mip.tolerances.mipgap.set(cplex_parameters['mipgap'])
        mip.parameters.mip.tolerances.absmipgap.set(cplex_parameters['absmipgap'])
        mip.parameters.mip.tolerances.integrality.set(cplex_parameters['integrality_tolerance'])

        # CPLEX Solution Pool Parameters
        mip.parameters.mip.limits.repairtries.set(cplex_parameters['repairtries'])
        mip.parameters.mip.pool.capacity.set(cplex_parameters['poolsize'])
        mip.parameters.mip.pool.replace.set(cplex_parameters['poolreplace'])
        # 0 = replace oldest /1: replace worst objective / #2 = replace least diverse solutions

    return mip


def add_mip_starts(mip, indices, pool, max_mip_starts=float('inf'), mip_start_effort_level=4):
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
    try:
        obj_cutoff = mip.parameters.mip.tolerances.uppercutoff.get()
    except:
        obj_cutoff = np.inf

    n_added = 0
    for k in range(0, len(pool)):
        if n_added < max_mip_starts:
            if pool.objvals[0] <= (obj_cutoff + np.finfo('float').eps):
                mip_start_name = "mip_start_" + str(n_added)
                mip_start_obj, _ = convert_to_risk_slim_cplex_solution(rho=pool.solutions[k, ], indices = indices, objval=pool.objvals[k])
                mip.MIP_starts.add(mip_start_obj, mip_start_effort_level, mip_start_name)
                n_added += 1
        else:
            break

    return mip


# Data-Related Computation
def load_loss_functions(Z,
                        loss_computation = "fast",
                        sample_weights = None,
                        rho_ub = None,
                        rho_lb = None,
                        L0_reg_ind= None,
                        L0_max=None):
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
    #todo load loss based on constraints
    #todo check to see if fast/lookup loss are installed

    ignore_sample_weights = (sample_weights is None) or (len(np.unique(sample_weights)) <= 1)


    if ignore_sample_weights:

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
            loss_value_tbl, prob_value_tbl, tbl_offset = lossfun.lookup_log_loss.get_loss_value_and_prob_tables(s_min,
                                                                                                                s_max)

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

    else:

        distinct_sample_weights = np.unique(sample_weights)
        Z = np.require(Z, requirements=['C'])
        total_sample_weights = np.sum(sample_weights)

        if len(distinct_sample_weights) == 2:
            print_log("USING WEIGHTED LOSS COMPUTATION WITH CLASS sample_weights (%1.4f, %1.4f)" % (
                distinct_sample_weights[0], distinct_sample_weights[1]))
        else:
            print_log("USING WEIGHTED LOSS COMPUTATION WITH SAMPLE sample_weights (TOTAL = %1.4f)" % (np.sum(total_sample_weights)))

        def compute_loss(rho):
            return lossfun.log_loss_weighted.log_loss_value(Z, sample_weights, total_sample_weights, rho)

        def compute_loss_cut(rho):
            return lossfun.log_loss_weighted.log_loss_value_and_slope(Z, sample_weights, total_sample_weights, rho)

        def compute_loss_from_scores(scores):
            return lossfun.log_loss_weighted.log_loss_value_from_scores(sample_weights, total_sample_weights, scores)

        compute_loss_real = compute_loss
        compute_loss_cut_real = compute_loss_cut
        compute_loss_from_scores_real = compute_loss_from_scores

    return (compute_loss,
            compute_loss_cut,
            compute_loss_from_scores,
            compute_loss_real,
            compute_loss_cut_real,
            compute_loss_from_scores_real)


# Scores-Based Bounds
def get_score_bounds(Z_min, Z_max, rho_lb, rho_ub, L0_reg_ind = None, L0_max = None):
    edge_values = np.vstack([Z_min * rho_lb,
                             Z_max * rho_lb,
                             Z_min * rho_ub,
                             Z_max * rho_ub])

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


def get_loss_bounds(Z, rho_ub, rho_lb, L0_reg_ind, L0_max=np.nan):
    # min value of loss = log(1+exp(-score)) occurs at max score for each point
    # max value of loss = loss(1+exp(-score)) occurs at min score for each point

    rho_lb = np.array(rho_lb)
    rho_ub = np.array(rho_ub)

    # get maximum number of regularized coefficients
    if np.isnan(L0_max):
        L0_max = Z.shape[0]

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
    s_min, s_max = get_score_bounds(Z_min=Z_min[variable_idx],
                                    Z_max=Z_max[variable_idx],
                                    rho_lb=coef_set.lb[variable_idx],
                                    rho_ub=coef_set.ub[variable_idx],
                                    L0_reg_ind=L0_reg_ind,
                                    L0_max=max_L0_value)

    # get max # of non-zero coefficients given model size limit
    conservative_offset = max(abs(s_min), abs(s_max)) + 1
    return conservative_offset


# Heuristics
def chained_updates(old_bounds, C_0_nnz, new_objval_at_feasible=None, new_objval_at_relaxation=None):

    new_bounds = dict(old_bounds)

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
    MAX_CHAIN_COUNT = 20  # defensive programming
    improved_bounds = True

    while improved_bounds and chain_count < MAX_CHAIN_COUNT:

        improved_bounds = False
        L0_penalty_min = np.sum(np.sort(C_0_nnz)[np.arange(0, int(new_bounds['L0_min']))])
        L0_penalty_max = np.sum(-np.sort(-C_0_nnz)[np.arange(0, int(new_bounds['L0_max']))])

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


def sequential_rounding(rho, Z, C_0, objval_cutoff=float('Inf')):
    """
    :param rho: continuous solution st. rho_lb[i] <= rho[i] <= rho_ub[i]
    :param objval_cutoff: cutoff value at which we return null
    :return: rho: integer solution st. rho_lb[i] <= rho[i] <= rho_ub[i]
    """

    P = rho.shape[0]
    dimensions_to_round = np.flatnonzero(np.mod(rho, 1)).tolist()
    floor_is_zero = np.equal(np.floor(rho), 0)
    ceil_is_zero = np.equal(np.ceil(rho), 0)

    dist_from_start_to_ceil = np.ceil(rho) - rho
    dist_from_start_to_floor = np.floor(rho) - rho

    scores = Z.dot(rho)
    best_objval = float('inf')
    early_stop_flag = False

    while len(dimensions_to_round) > 0:

        objvals_at_floor = np.array([np.nan] * P)
        objvals_at_ceil = np.array([np.nan] * P)
        current_penalty = get_L0_penalty(rho)

        for dim_idx in dimensions_to_round:

            # scores go from center to ceil -> center + dist_from_start_to_ceil
            scores += dist_from_start_to_ceil[dim_idx] * Z[:, dim_idx]
            objvals_at_ceil[dim_idx] = compute_loss_from_scores_real(scores)

            # move from ceil to floor => -1*Z_j
            scores -= Z[:, dim_idx]
            objvals_at_floor[dim_idx] = compute_loss_from_scores_real(scores)

            # scores go from floor to center -> floor - dist_from_start_to_floor
            scores -= dist_from_start_to_floor[dim_idx] * Z[:, dim_idx]
            # assert(np.all(np.isclose(scores, base_scores)))

            if ceil_is_zero[dim_idx]:
                objvals_at_ceil[dim_idx] -= C_0[dim_idx]
            elif floor_is_zero[dim_idx]:
                objvals_at_floor[dim_idx] -= C_0[dim_idx]

        # adjust for penalty value
        objvals_at_ceil += current_penalty
        objvals_at_floor += current_penalty

        best_objval_at_ceil = np.nanmin(objvals_at_ceil)
        best_objval_at_floor = np.nanmin(objvals_at_floor)
        best_objval = min(best_objval_at_ceil, best_objval_at_floor)

        if best_objval > objval_cutoff:
            best_objval = float('nan')
            early_stop_flag = True
            break
        else:
            if best_objval_at_ceil <= best_objval_at_floor:
                best_dim = np.nanargmin(objvals_at_ceil)
                rho[best_dim] += dist_from_start_to_ceil[best_dim]
                scores += dist_from_start_to_ceil[best_dim] * Z[:, best_dim]
            else:
                best_dim = np.nanargmin(objvals_at_floor)
                rho[best_dim] += dist_from_start_to_floor[best_dim]
                scores += dist_from_start_to_floor[best_dim] * Z[:, best_dim]

        # assert(np.all(np.isclose(scores, Z.dot(rho))))
        dimensions_to_round.remove(best_dim)

    return rho, best_objval, early_stop_flag


def discrete_descent(rho, Z, C_0, rho_ub, rho_lb, descent_dimensions=None, print_flag=False):
    """
    given a initial feasible solution, rho, produces an improved solution that is 1-OPT
    (i.e. the objective value does not decrease by moving in any single dimension)
    at each iteration, the algorithm moves in the dimension that yields the greatest decrease in objective value
    the best step size is each dimension is computed using a directional search strategy that saves computation
    """
    #print_flag = False

    # initialize key variables
    MAX_ITERATIONS = 500
    MIN_IMPROVEMENT_PER_STEP = float(1e-10)
    n_iterations = 0
    P = rho.shape[0]
    rho = np.require(np.require(rho, dtype=np.int_), dtype=np.float_)
    if descent_dimensions is None:
        descent_dimensions = range(0, P)

    search_dimensions = descent_dimensions
    base_scores = Z.dot(rho)
    base_loss = compute_loss_from_scores(base_scores)
    base_objval = base_loss + get_L0_penalty(rho)
    keep_searching = True

    while keep_searching and n_iterations < MAX_ITERATIONS:

        # compute the best objective value / step size in each dimension
        best_objval_by_dim = np.array([np.nan] * P)
        best_coef_by_dim = np.array([np.nan] * P)

        for k in search_dimensions:
            feasible_coefs_for_dim = np.arange(int(rho_lb[k]), int(rho_ub[k]) + 1)  # TODO CHANGE THIS
            objvals = compute_objvals_at_dim(k, feasible_coefs_for_dim, rho, base_scores, base_loss, C_0)
            objvals[np.where(objvals == base_objval)] = np.inf
            best_objval_by_dim[k] = np.nanmin(objvals)
            best_coef_by_dim[k] = feasible_coefs_for_dim[np.nanargmin(objvals)]

        # check if there exists a move that yields an improvement
        # print_log('')
        # print_log('ITERATION %d' % n_iterations)
        # print_log('search dimensions has %d/%d dimensions' % (len(search_dimensions), P))
        # print_log(search_dimensions)
        # print_log('best_objval_by_dim')
        # print_log(["{0:0.20f}".format(i) for i in best_objval_by_dim])

        next_objval = np.nanmin(best_objval_by_dim)
        # print_log('IMPROVEMENT: %1.20f' % (base_objval - next_objval))

        if next_objval < (base_objval - MIN_IMPROVEMENT_PER_STEP):
            # take the best step in the best direction
            step_dim = int(np.nanargmin(best_objval_by_dim))

            # if print_flag:
            #     print_log("improving objective value from %1.16f to %1.16f" % (base_objval, next_objval))
            #     print_log(
            #         "changing rho[%d] from %1.0f to %1.0f" % (step_dim, rho[step_dim], best_coef_by_dim[step_dim]))

            rho[step_dim] = best_coef_by_dim[step_dim]

            # recompute base objective value/loss/scores
            base_objval = next_objval
            base_loss = base_objval - get_L0_penalty(rho)
            base_scores = Z.dot(rho)

            # remove the current best direction from the set of directions to explore
            search_dimensions = descent_dimensions
            search_dimensions.remove(step_dim)
            n_iterations += 1
        else:
            keep_searching = False

    # if print_flag:
    #     print_log("completed %d iterations" % n_iterations)
    #     print_log("current: %1.10f < best possible: %1.10f" % (base_objval, next_objval))

    return rho, base_loss, base_objval


def compute_objvals_at_dim(dim_index,
                           feasible_coef_values,
                           base_rho,
                           base_scores,
                           base_loss,
                           C_0):
    """
    finds the value of rho[j] in feasible_coef_values that minimizes log_loss(rho) + C_0j
    :param dim_index:
    :param feasible_coef_values:
    :param base_rho:
    :param base_scores:
    :param base_loss:
    :param C_0:
    :return:
    """

    # copy stuff because ctypes
    scores = np.copy(base_scores)

    # initialize parameters
    P = base_rho.shape[0]
    base_coef_value = base_rho[dim_index]
    base_index = np.where(feasible_coef_values == base_coef_value)[0]
    loss_at_coef_value = np.array([np.nan] * len(feasible_coef_values))
    loss_at_coef_value[base_index] = np.float(base_loss)
    Z_dim = Z[:, dim_index]

    # start by moving forward
    forward_indices = np.where(base_coef_value <= feasible_coef_values)[0]
    forward_step_sizes = np.diff(feasible_coef_values[forward_indices] - base_coef_value)
    n_forward_steps = len(forward_step_sizes)
    stop_after_first_forward_step = False

    best_loss = base_loss
    total_distance_from_base = 0

    for i in range(0, n_forward_steps):
        scores += forward_step_sizes[i] * Z_dim
        total_distance_from_base += forward_step_sizes[i]
        current_loss = compute_loss_from_scores(scores)
        if current_loss >= best_loss:
            stop_after_first_forward_step = (i == 0)
            break
        loss_at_coef_value[forward_indices[i + 1]] = current_loss
        best_loss = current_loss

    # if the first step forward didn't lead to a decrease in loss, then move backwards
    move_backward = stop_after_first_forward_step or n_forward_steps == 0

    if move_backward:

        # compute backward steps
        backward_indices = np.flipud(np.where(feasible_coef_values <= base_coef_value)[0])
        backward_step_sizes = np.diff(feasible_coef_values[backward_indices] - base_coef_value)
        n_backward_steps = len(backward_step_sizes)

        # correct size of first backward step if you took 1 step forward
        if n_backward_steps > 0 and n_forward_steps > 0:
            backward_step_sizes[0] = backward_step_sizes[0] - forward_step_sizes[0]

        best_loss = base_loss

        for i in range(0, n_backward_steps):
            scores += backward_step_sizes[i] * Z_dim
            total_distance_from_base += backward_step_sizes[i]
            current_loss = compute_loss_from_scores(scores)
            if current_loss >= best_loss:
                break
            loss_at_coef_value[backward_indices[i + 1]] = current_loss
            best_loss = current_loss

    # at this point scores == base_scores + step_distance*Z_dim
    # assert(all(np.isclose(scores, base_scores + total_distance_from_base * Z_dim)))

    # compute objective values by adding penalty values to all other indices
    other_dim_idx = np.where(dim_index != np.arange(0, P))[0]
    other_dim_penalty = np.sum(C_0[other_dim_idx] * (base_rho[other_dim_idx] != 0))
    objval_at_coef_values = loss_at_coef_value + other_dim_penalty

    if C_0[dim_index] > 0.0:

        # increase objective value at every non-zero coefficient value by C_0j
        nonzero_coef_idx = np.flatnonzero(feasible_coef_values)
        objval_at_coef_values[nonzero_coef_idx] = objval_at_coef_values[nonzero_coef_idx] + C_0[dim_index]

        # compute value at coef[j] == 0 if needed
        zero_coef_idx = np.where(feasible_coef_values == 0)[0]
        if np.isnan(objval_at_coef_values[zero_coef_idx]):
            # steps_from_here_to_zero: step_from_here_to_base + step_from_base_to_zero
            # steps_from_here_to_zero: -step_from_base_to_here + -step_from_zero_to_base
            steps_to_zero = -(base_coef_value + total_distance_from_base)
            scores += steps_to_zero * Z_dim
            objval_at_coef_values[zero_coef_idx] = compute_loss_from_scores(scores) + other_dim_penalty
            # assert(all(np.isclose(scores, base_scores - base_coef_value * Z_dim)))

    # return objective value at feasible coefficients
    return objval_at_coef_values


# Initialization Procedure
def initialize_lattice_cpa(risk_slim_settings, warmstart_settings, cplex_parameters, compute_loss_real, compute_loss_cut_real, bounds = None):
    """

    Returns
    -------
    cuts
    solution pool
    bounds

    """
    warmstart_settings = dict(warmstart_settings)
    risk_slim_settings = dict(risk_slim_settings)
    #get_objval
    #check_feasible


    C_0 = np.array(risk_slim_settings['coef_set'].C_0j)
    L0_reg_ind = np.isnan(C_0)
    C_0[L0_reg_ind] = risk_slim_settings['C_0']
    C_0_nnz = C_0[L0_reg_ind]

    if bounds is None:
        bounds = {
            'objval_min': 0.0,
            'objval_max': CPX_INFINITY,
            'loss_min': 0.0,
            'loss_max': CPX_INFINITY,
            'L0_min': 0,
            'L0_max': risk_slim_settings['L0_max'],
        }
    warmstart_settings['type'] = 'cvx'
    risk_slim_settings.update(bounds)
    risk_slim_settings['relax_integer_variables'] = True

    #create RiskSLIM LP
    risk_slim_lp, risk_slim_lp_indices = create_risk_slim(risk_slim_settings)
    risk_slim_lp = set_cplex_mip_parameters(risk_slim_lp, cplex_parameters, display_cplex_progress = warmstart_settings['display_cplex_progress'])

    # solve risk_slim_lp LP using standard CPA
    cpa_stats, initial_cuts, cts_pool = cutting_plane_algorithm(risk_slim_lp,
                                                                risk_slim_lp_indices,
                                                                warmstart_settings,
                                                                compute_loss_real,
                                                                compute_loss_cut_real)

    # update bounds
    initial_bounds = chained_updates(bounds, C_0_nnz, new_objval_at_relaxation=cpa_stats['lowerbound'])
    initial_pool = SolutionPool(cts_pool.P)

    #remove redundant solutions, remove infeasible solutions, order solutions by objective value of RiskSLIMLP
    cts_pool = cts_pool.distinct().removeInfeasible(check_feasible).sort()

    if warmstart_settings['use_sequential_rounding']:
        initial_pool, _, _ = sequential_round_solution_pool(cts_pool,
                                                            max_runtime=warmstart_settings['sequential_rounding_max_runtime'],
                                                            max_solutions=warmstart_settings['sequential_rounding_max_solutions'],
                                                            objval_cutoff=bounds['objval_max'],
                                                            L0_min=bounds['L0_min'],
                                                            L0_max=bounds['L0_max'])

        initial_pool = initial_pool.distinct().sort()
        bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible=np.min(initial_pool.objvals))
    else:
        initial_pool, _, _ = round_solution_pool(cts_pool, constraints)

    initial_pool.computeObjvals(get_objval)
    if warmstart_settings['polishing_after'] and len(initial_pool) > 0:
        initial_pool, _, _ = discrete_descent_solution_pool(initial_pool,
                                                            warmstart_settings['polishing_max_runtime'],
                                                            warmstart_settings['polishing_max_solutions'])

        initial_pool = initial_pool.removeInfeasible(check_feasible).distinct().sort()

    if len(initial_pool) > 0:
        initial_bounds = chained_updates(bounds, C_0_nnz, new_objval_at_feasible=np.min(initial_pool.objvals))

    return initial_pool, initial_cuts, initial_pool


def cutting_plane_algorithm(MIP, indices, settings, compute_loss, compute_loss_cut):

    settings = get_or_set_default(settings, 'print_flag', True)

    if settings['print_flag']:
        def print_from_function(msg):
            print_log(msg)
    else:
        def print_from_function(msg):
            pass

    settings = get_or_set_default(settings, 'type', 'cvx')
    settings = get_or_set_default(settings, 'update_bounds', True)
    settings = get_or_set_default(settings, 'display_progress', True)
    settings = get_or_set_default(settings, 'save_progress', True)
    settings = get_or_set_default(settings, 'max_tolerance', 0.00001)
    settings = get_or_set_default(settings, 'max_iterations', 10000)
    settings = get_or_set_default(settings, 'max_runtime', 100.0)
    settings = get_or_set_default(settings, 'max_runtime_per_iteration', 10000.0)
    settings = get_or_set_default(settings, 'max_cplex_time_per_iteration', 60.0)

    rho_idx = indices["rho"]
    loss_idx = indices["loss"]
    alpha_idx = indices["alpha"]
    cut_idx = loss_idx + rho_idx
    objval_idx = indices["objval"]
    L0_idx = indices["L0_norm"]

    if len(alpha_idx) == 0:
        def get_alpha():
            return np.array([])
    else:
        def get_alpha():
            return np.array(MIP.solution.get_values(alpha_idx))

    if type(loss_idx) is list and len(loss_idx) == 1:
        loss_idx = loss_idx[0]

    C_0_alpha = indices['C_0_alpha']
    nnz_ind = np.flatnonzero(C_0_alpha)
    C_0_nnz = C_0_alpha[nnz_ind]

    #setup variables for updating bounds
    if settings['update_bounds']:
        bounds = {
            'loss_min': MIP.variables.get_lower_bounds(loss_idx),
            'loss_max': MIP.variables.get_upper_bounds(loss_idx),
            'objval_min': MIP.variables.get_lower_bounds(objval_idx),
            'objval_max': MIP.variables.get_upper_bounds(objval_idx),
            'L0_min': MIP.variables.get_lower_bounds(L0_idx),
            'L0_max': MIP.variables.get_upper_bounds(L0_idx),
        }
        if settings['type'] is 'cvx':
            vtypes = 'C'
            def update_problem_bounds(bounds, lb, ub):
                return chained_updates_for_lp(bounds,
                                              C_0_nnz,
                                              new_objval_at_feasible = ub,
                                              new_objval_at_relaxation = lb,
                                              print_flag = False)
        elif settings['type'] is 'ntree':
            vtypes = MIP.variables.get_types(rho_idx)
            def update_problem_bounds(bounds, lb, ub):
                return chained_updates(bounds, C_0_nnz,
                                       new_objval_at_feasible = ub,
                                       new_objval_at_relaxation = lb,
                                       print_flag = False)

    objval = 0.0
    upperbound = CPX_INFINITY
    lowerbound = 0.0
    solutions = []
    objvals = []
    upperbounds = []
    lowerbounds = []
    simplex_iterations = []
    cut_times = []
    total_times = []
    n_iterations = 0
    simplex_iteration = 0
    stop_reason = 'aborted:reached_max_cuts'

    max_runtime = settings['max_runtime']
    remaining_total_time = max_runtime
    run_start_time = time.time()

    while n_iterations < settings['max_iterations']:

        iteration_start_time = time.time()
        current_timelimit = min(remaining_total_time, settings['max_cplex_time_per_iteration'])
        MIP.parameters.timelimit.set(float(current_timelimit))
        MIP.solve()
        solution_status = MIP.solution.status[MIP.solution.get_status()]

        # get solution
        if solution_status in ('optimal', 'optimal_tolerance', 'MIP_optimal'):
            rho = np.array(MIP.solution.get_values(rho_idx))
            alpha = get_alpha()
            simplex_iteration = int(MIP.solution.progress.get_num_iterations())
        else:
            stop_reason = solution_status
            print_from_function('BREAKING NTREE CP LOOP NON-OPTIMAL SOLUTION FOUND: %s' % solution_status)
            break

        # compute cut
        cut_start_time = time.time()
        loss_value, loss_slope = compute_loss_cut(rho)
        cut_lhs = [float(loss_value - loss_slope.dot(rho))]
        cut_constraint = [cplex.SparsePair(ind = cut_idx, val = [1.0] + (-loss_slope).tolist())]
        cut_name = ["ntree_cut_%d" % n_iterations]
        cut_time = time.time() - cut_start_time

        # compute objective bounds
        objval = float(loss_value + alpha.dot(C_0_alpha))
        upperbound = min(upperbound, objval)
        lowerbound = MIP.solution.get_objective_value()
        relative_gap = (upperbound - lowerbound)/(upperbound + np.finfo('float').eps)

        #update variable bounds
        if settings['update_bounds']:
            bounds = update_problem_bounds(bounds, lb = lowerbound, ub = upperbound)

        # update run stats
        n_iterations += 1
        current_time = time.time()
        total_time = current_time - run_start_time
        iteration_time = current_time - iteration_start_time
        cplex_time = iteration_time - cut_time
        remaining_total_time  = max(max_runtime - total_time, 0.0)

        # print information
        if settings['display_progress']:
            print_update(rho, n_iterations, upperbound, lowerbound, relative_gap, vtypes)

        #save progress
        if settings['save_progress']:
            solutions.append(rho)
            objvals.append(objval)
            upperbounds.append(upperbound)
            lowerbounds.append(lowerbound)
            total_times.append(total_time)
            cut_times.append(cut_time)
            simplex_iterations.append(simplex_iteration)

        # check termination conditions
        if relative_gap < settings['max_tolerance']:
            stop_reason = 'converged:gap_within_tolerance'
            print_from_function('BREAKING NTREE CP LOOP - MAX TOLERANCE')
            break

        if cplex_time > settings['max_cplex_time_per_iteration']:
            stop_reason = 'aborted:reached_max_train_time'
            break

        if iteration_time > settings['max_runtime_per_iteration']:
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('BREAKING NTREE CP LOOP - REACHED MAX RUNTIME PER ITERATION')
            break

        if (total_time > settings['max_runtime']) or (remaining_total_time == 0.0):
            stop_reason = 'aborted:reached_max_train_time'
            print_from_function('BREAKING NTREE CP LOOP - REACHED MAX RUNTIME')
            break

        # switch bounds
        if settings['update_bounds']:
            MIP.variables.set_lower_bounds(L0_idx, bounds['L0_min'])
            MIP.variables.set_upper_bounds(L0_idx, bounds['L0_max'])
            MIP.variables.set_lower_bounds(loss_idx, bounds['loss_min'])
            MIP.variables.set_upper_bounds(loss_idx, bounds['loss_max'])
            MIP.variables.set_lower_bounds(objval_idx, bounds['objval_min'])
            MIP.variables.set_upper_bounds(objval_idx, bounds['objval_max'])

        # add loss cut
        MIP.linear_constraints.add(lin_expr = cut_constraint,
                                   senses = ["G"],
                                   rhs = cut_lhs,
                                   names = cut_name)

    #collect stats
    stats = {
        'solution': rho,
        'stop_reason': stop_reason,
        'n_iterations': n_iterations,
        'simplex_iteration': simplex_iteration,
        'objval': objval,
        'upperbound': upperbound,
        'lowerbound': lowerbound,
        'cut_time': cut_time,
        'total_time': total_time,
        'cplex_time': total_time - cut_time,
    }
    if settings['update_bounds']:
        stats.update(bounds)

    if settings['save_progress']:
        stats['lowerbounds'] = lowerbounds
        stats['upperbounds'] = upperbounds
        stats['objvals'] = objvals
        stats['simplex_iterations'] = simplex_iterations
        stats['solutions'] = solutions
        stats['cut_times'] = cut_times
        stats['total_times'] = total_times
        stats['cplex_times'] = [total_times[i] - cut_times[i] for i in range(0, len(total_times))]

    #collect cuts
    idx = range(indices['n_constraints'], MIP.linear_constraints.get_num(), 1)
    cuts = {
        'coefs': MIP.linear_constraints.get_rows(idx),
        'lhs': MIP.linear_constraints.get_rhs(idx)
    }

    #create solution pool
    pool = {
        'solutions': [],
        'objvals': []
    }

    if settings['type'] is 'ntree':
        C_0_rho = np.array(indices['C_0_rho'])
        #explicitly compute objective value of each solution in solution pool
        for i in range(0, MIP.solution.pool.get_num()):
            rho = np.array(MIP.solution.pool.get_values(i, rho_idx))
            objval = compute_loss(rho) + np.sum(C_0_rho[np.flatnonzero(np.array(rho))])
            pool['solutions'].append(rho)
            pool['objvals'].append(objval)
    else:
        for i in range(0, MIP.solution.pool.get_num()):
            rho = np.array(MIP.solution.pool.get_values(i, rho_idx))
            objval = MIP.solution.pool.get_objective_value(i)
            pool['solutions'].append(rho)
            pool['objvals'].append(objval)

    pool['solutions'] += solutions
    pool['objvals'] += objvals

    return stats, cuts, pool


def chained_updates_for_lp(old_bounds, C_0_nnz, new_objval_at_feasible=None, new_objval_at_relaxation=None):
    new_bounds = dict(old_bounds)

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
    MAX_CHAIN_COUNT = 20
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


# todo: add methods to SolutionPool so as to run the following as filter-map
def round_solution_pool(pool, constraints):

    pool.distinct().sort()
    P = pool.P
    L0_reg_ind = np.isnan(constraints['coef_set'].C_0j)
    L0_max = constraints['L0_max']
    rounded_pool = SolutionPool(P)

    for solution in pool.solutions:
        # sort from largest to smallest coefficients
        feature_order = np.argsort([-abs(x) for x in solution])
        rounded_solution = np.zeros(shape=(1, P))
        l0_norm_count = 0
        for k in range(0, P):
            j = feature_order[k]
            if not L0_reg_ind[j]:
                rounded_solution[0, j] = np.round(solution[j], 0)
            elif l0_norm_count < L0_max:
                rounded_solution[0, j] = np.round(solution[j], 0)
                l0_norm_count += L0_reg_ind[j]

        rounded_pool.add(objvals=np.nan, solutions=rounded_solution)

    rounded_pool.distinct().sort()
    return rounded_pool


def sequential_round_solution_pool(pool,
                                   Z,
                                   C_0,
                                   max_runtime=float('inf'),
                                   max_solutions=float('inf'),
                                   objval_cutoff=float('inf'),
                                   L0_min=0,
                                   L0_max=float('inf')):
    """

    Parameters
    ----------
    pool
    Z
    C_0
    max_runtime
    max_solutions
    objval_cutoff
    L0_min
    L0_max

    Returns
    -------

    """
    # quick return
    if len(pool) == 0:
        return pool, 0.0, 0

    P = pool.P
    total_runtime = 0.0

    #todo: filter out solutions that can only be rounded one way
    #rho is integer

    #if model size constraint is non-trivial, remove solutions that will violate the model size constraint
    if L0_min > 0 and L0_max < P:
        L0_reg_ind = np.isnan(C_0)

        def rounded_model_size_is_ok(rho):
            abs_rho = abs(rho)
            rounded_rho_l0_min = np.count_nonzero(np.floor(abs_rho[L0_reg_ind]))
            rounded_rho_l0_max = np.count_nonzero(np.ceil(abs_rho[L0_reg_ind]))
            return rounded_rho_l0_max >= L0_min and rounded_rho_l0_min <= L0_max

        pool = pool.removeInfeasible(rounded_model_size_is_ok)

    pool = pool.sort()
    n_to_round = min(max_solutions, len(pool))
    rounded_pool = SolutionPool(pool.P)

    #round solutions using sequential rounding
    for n in range(n_to_round):

        start_time = time.time()
        solution, objval, early_stop = sequential_rounding(pool.solutions[n], Z, C_0, objval_cutoff)
        total_runtime += time.time() - start_time

        if not early_stop:
            rounded_pool = rounded_pool.add(objvals=objval, solutions=solution)

        if total_runtime > max_runtime:
            break

    return rounded_pool, total_runtime, len(rounded_pool)

def discrete_descent_solution_pool(pool,
                                   Z,
                                   C_0,
                                   constraints,
                                   max_runtime=float('inf'),
                                   max_solutions=float('inf')):
    """
    runs DCD polishing for all solutions in the a solution pool
    can be stopped early using max_runtime or max_solutions

    Parameters
    ----------
    pool
    max_runtime
    max_solutions

    Returns
    -------
    new solution pool, total polishing time, and # of solutions polished

    """

    # quick return
    if len(pool) == 0:
        return pool, 0.0, 0

    pool = pool.sort()
    polished_pool = SolutionPool(pool.P)
    n_to_polish = min(max_solutions, len(pool))
    total_runtime = 0.0
    rho_ub = np.array(constraints['coef_set'].ub)
    rho_lb = np.array(constraints['coef_set'].lb)

    for n in range(n_to_polish):

        start_time = time.time()
        polished_solution, _, polished_objval = discrete_descent(pool.solutions[n], Z, C_0, rho_lb, rho_ub)
        total_runtime += time.time() - start_time

        polished_pool = polished_pool.add(objvals=polished_objval, solutions=polished_solution)

        if total_runtime > max_runtime:
            break

    n_polished = len(polished_pool)
    polished_pool = polished_pool.append(pool).sort()
    return polished_pool, total_runtime, n_polished