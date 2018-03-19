import time
import numpy as np
from cplex.exceptions import CplexError
from cplex.callbacks import LazyConstraintCallback, HeuristicCallback
from .mip import *
from .helper_functions import print_log, is_integer, cast_to_integer
from .setup_functions import get_loss_bounds, setup_loss_functions, setup_training_weights, setup_penalty_parameters, setup_objective_functions
from .solution_classes import SolutionQueue, SolutionPool
from .bound_tightening import chained_updates
from .heuristics import discrete_descent, sequential_rounding
from .initialization import initialize_lattice_cpa
from .default_settings import DEFAULT_LCPA_SETTINGS



DEFAULT_BOUNDS = {
    'objval_min': 0.0,
    'objval_max': float('inf'),
    'loss_min': 0.0,
    'loss_max': float('inf'),
    'L0_min': 0,
    'L0_max': float('inf'),
    }


def run_lattice_cpa(data, constraints, settings = DEFAULT_LCPA_SETTINGS):

    # initialize MIP for lattice CPA
    mip_objects = setup_lattice_cpa(data, constraints, settings)

    # solve MIP using lattice CPA
    model_info, mip_info, lcpa_info = finish_lattice_cpa(data, constraints, mip_objects, settings)

    return model_info, mip_info, lcpa_info


def setup_lattice_cpa(data, constraints, settings = DEFAULT_LCPA_SETTINGS):
    """

    Parameters
    ----------
    data, dict containing training data should pass check_data
    constraints, dict containing 'L0_min, L0_max, CoefficientSet'
    settings

    Returns
    -------
    mip_objects 
    
    """
    settings = dict(settings)
    settings = {key: settings[key] if key in settings else DEFAULT_LCPA_SETTINGS[key] for key in DEFAULT_LCPA_SETTINGS}

    # separate settings into more manageable objects
    init_settings = {key.lstrip('init_'): settings[key] for key in settings if key.startswith('init_')}
    cplex_settings = {key.lstrip('cplex_'): settings[key] for key in settings if key.startswith('cplex_')}
    lcpa_settings = {key: settings[key] for key in settings if settings if
                     not (key.startswith('init_') or key.startswith('cplex_'))}

    # data
    Z = data['X'] * data['Y']
    N, P = Z.shape

    # sample weights and case weights
    training_weights = setup_training_weights(Y = data['Y'], sample_weights = data['sample_weights'],
                                              w_pos = settings['w_pos'])

    # trade-off parameters
    c0_value, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(c0_value = lcpa_settings['c0_value'],
                                                                  coef_set = constraints['coef_set'])

    # constraints
    rho_lb = np.array(constraints['coef_set'].lb)
    rho_ub = np.array(constraints['coef_set'].ub)
    L0_min = constraints['L0_min']
    L0_max = constraints['L0_max']

    # loss functions
    (compute_loss,
     compute_loss_cut,
     compute_loss_from_scores,
     compute_loss_real,
     compute_loss_cut_real,
     compute_loss_from_scores_real) = setup_loss_functions(Z,
                                                           sample_weights = training_weights,
                                                           loss_computation = settings['loss_computation'],
                                                           rho_ub = constraints['coef_set'].ub,
                                                           rho_lb = constraints['coef_set'].lb,
                                                           L0_reg_ind = np.isnan(constraints['coef_set'].C_0j),
                                                           L0_max = constraints['L0_max'])

    # major components
    (get_objval,
     get_L0_norm,
     get_L0_penalty,
     get_alpha,
     get_L0_penalty_from_alpha) = setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz)

    def is_feasible(rho, L0_min = L0_min, L0_max = L0_max, rho_lb = rho_lb, rho_ub = rho_ub):
        return np.all(rho_ub >= rho) and np.all(rho_lb <= rho) and (
                L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max)

    # compute bounds on objective value
    bounds = dict(DEFAULT_BOUNDS)
    bounds['L0_min'] = constraints['L0_min']
    bounds['L0_max'] = constraints['L0_max']
    bounds['loss_min'], bounds['loss_max'] = get_loss_bounds(Z, rho_ub, rho_lb, L0_reg_ind, L0_max)

    # initialize
    initial_pool = SolutionPool(P)
    initial_cuts = None

    # check if trivial solution is feasible, if so add it to the pool and update bounds
    trivial_solution = np.zeros(P)
    if is_feasible(trivial_solution):
        trivial_objval = compute_loss(trivial_solution)
        bounds['objval_max'] = min(bounds['objval_max'], trivial_objval)
        bounds['loss_max'] = min(bounds['loss_max'], trivial_objval)
        bounds = chained_updates(bounds, C_0_nnz)
        initial_pool = initial_pool.add(objvals = trivial_objval, solutions = trivial_solution)

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
        initial_pool, initial_cuts, initial_bounds = initialize_lattice_cpa(Z = Z,
                                                                            c0_value = lcpa_settings['c0_value'],
                                                                            constraints = constraints,
                                                                            bounds = bounds,
                                                                            settings = init_settings,
                                                                            risk_slim_settings = risk_slim_settings,
                                                                            cplex_settings = cplex_settings,
                                                                            compute_loss_from_scores = compute_loss_from_scores,
                                                                            compute_loss_real = compute_loss_real,
                                                                            compute_loss_cut_real = compute_loss_cut_real,
                                                                            compute_loss_from_scores_real = compute_loss_from_scores_real,
                                                                            get_objval = get_objval,
                                                                            get_L0_penalty = get_L0_penalty,
                                                                            is_feasible = is_feasible)

        bounds.update(initial_bounds)
        risk_slim_settings.update(initial_bounds)

    # create risk_slim mip
    risk_slim_mip, risk_slim_indices = create_risk_slim(risk_slim_settings)
    risk_slim_mip = set_cplex_mip_parameters(risk_slim_mip, cplex_settings,
                                             display_cplex_progress = lcpa_settings['display_cplex_progress'])
    risk_slim_mip.parameters.timelimit.set(lcpa_settings['max_runtime'])
    risk_slim_indices['C_0_nnz'] = C_0_nnz
    risk_slim_indices['L0_reg_ind'] = L0_reg_ind

    # mip
    mip_objects = {
        'mip': risk_slim_mip,
        'indices':risk_slim_indices,
        'bounds': bounds,
        'initial_pool': initial_pool,
        'initial_cuts': initial_cuts,
        }

    return mip_objects


def finish_lattice_cpa(data, constraints, mip_objects, settings = DEFAULT_LCPA_SETTINGS):
    """

    Parameters
    ----------
    data, dict containing training data should pass check_data
    constraints, dict containing 'L0_min, L0_max, CoefficientSet'
    settings
    mip_objects output of setup_risk_slim
    
    Returns
    ------- 

    """
    settings = dict(settings)
    settings = {key: settings[key] if key in settings else DEFAULT_LCPA_SETTINGS[key] for key in DEFAULT_LCPA_SETTINGS}

    # separate settings into more manageable objects
    cplex_settings = {key.lstrip('cplex_'): settings[key] for key in settings if key.startswith('cplex_')}
    lcpa_settings = {key: settings[key] for key in settings if settings if not (key.startswith('init_') or key.startswith('cplex_'))}

    # data
    Z = data['X'] * data['Y']
    N, P = Z.shape

    # unpack variables from setup_risk_slim
    risk_slim_mip = mip_objects['mip']
    indices = mip_objects['indices']
    bounds = mip_objects['bounds']
    initial_pool = mip_objects['initial_pool']
    initial_cuts = mip_objects['initial_cuts']

    # sample weights and case weights
    training_weights = setup_training_weights(Y = data['Y'], sample_weights = data['sample_weights'], w_pos = settings['w_pos'])

    # trade-off parameter
    c0_value, C_0, L0_reg_ind, C_0_nnz = setup_penalty_parameters(c0_value = lcpa_settings['c0_value'], coef_set = constraints['coef_set'])

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
     compute_loss_from_scores_real) = setup_loss_functions(Z,
                                                           sample_weights = training_weights,
                                                           loss_computation = settings['loss_computation'],
                                                           rho_ub = constraints['coef_set'].ub,
                                                           rho_lb = constraints['coef_set'].lb,
                                                           L0_reg_ind = np.isnan(constraints['coef_set'].C_0j),
                                                           L0_max = constraints['L0_max'])

    # setup function handles for key functions
    # major components
    (get_objval,
     get_L0_norm,
     get_L0_penalty,
     get_alpha,
     get_L0_penalty_from_alpha) = setup_objective_functions(compute_loss, L0_reg_ind, C_0_nnz)

    def is_feasible(rho, L0_min = L0_min, L0_max = L0_max, rho_lb = rho_lb, rho_ub = rho_ub):
        return np.all(rho_ub >= rho) and np.all(rho_lb <= rho) and (L0_min <= np.count_nonzero(rho[L0_reg_ind]) <= L0_max)

    risk_slim_mip = set_cplex_mip_parameters(risk_slim_mip, cplex_settings, display_cplex_progress = lcpa_settings['display_cplex_progress'])
    risk_slim_mip.parameters.timelimit.set(lcpa_settings['max_runtime'])

    # setup callback functions
    control = {
        'incumbent': np.nan * np.zeros(P),
        'upperbound': float('inf'),
        'bounds': dict(bounds),
        'lowerbound': 0.0,
        'relative_gap': float('inf'),
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
                           compute_loss_cut = compute_loss_cut,
                           get_alpha = get_alpha,
                           get_L0_penalty_from_alpha = get_L0_penalty_from_alpha,
                           initial_cuts = initial_cuts,
                           cut_queue = lcpa_cut_queue,
                           polish_queue = lcpa_polish_queue)

        heuristic_cb = risk_slim_mip.register_callback(PolishAndRoundCallback)
        heuristic_cb.initialize(indices = indices,
                                control = control,
                                settings = lcpa_settings,
                                cut_queue = lcpa_cut_queue,
                                polish_queue = lcpa_polish_queue,
                                get_objval = get_objval,
                                get_L0_norm = get_L0_norm,
                                is_feasible = is_feasible,
                                dcd_polishing = lambda rho: discrete_descent(rho, Z, C_0, rho_ub, rho_lb, get_L0_penalty, compute_loss_from_scores),
                                sequential_rounding = lambda rho, cutoff: sequential_rounding(rho, Z, C_0, compute_loss_from_scores_real, get_L0_penalty, cutoff))

    else:
        loss_cb = risk_slim_mip.register_callback(LossCallback)
        loss_cb.initialize(indices = indices,
                           control = control,
                           settings = lcpa_settings,
                           compute_loss_cut = compute_loss_cut,
                           get_alpha = get_alpha,
                           get_L0_penalty_from_alpha = get_L0_penalty_from_alpha,
                           initial_cuts = initial_cuts)

    # attach solution pool
    if len(initial_pool) > 0:
        if lcpa_settings['polish_flag']:
            lcpa_polish_queue.add(initial_pool.objvals[0], initial_pool.solutions[0])
            # we initialize using the polish_queue when possible since the CPLEX MIPStart interface is tricky
        else:
            risk_slim_mip = add_mip_starts(risk_slim_mip, indices, initial_pool, mip_start_effort_level=risk_slim_mip.MIP_starts.effort_level.repair)

        if lcpa_settings['add_cuts_at_heuristic_solutions'] and len(initial_pool) > 1:
            lcpa_cut_queue.add(initial_pool.objvals[1:], initial_pool.solutions[1:])

    # solve using lcpa
    control['start_time'] = time.time()
    risk_slim_mip.solve()
    control['total_run_time'] = time.time() - control['start_time']
    control.pop('start_time')

    # record mip solution statistics
    try:
        control['incumbent'] = np.array(risk_slim_mip.solution.get_values(indices['rho']))
        control['upperbound'] = risk_slim_mip.solution.get_objective_value()
        control['lowerbound'] = risk_slim_mip.solution.MIP.get_best_objective()
        try:
            control['relative_gap'] = risk_slim_mip.solution.MIP.get_mip_relative_gap()
        except CplexError:
            control['relative_gap'] = 1.0 - (control['lowerbound'] / (control['upperbound'] + np.finfo('float').eps))
        control['found_solution'] = True
    except CplexError:
        control['found_solution'] = False

    control['cplex_status'] = risk_slim_mip.solution.get_status_string()
    control['total_callback_time'] = control['total_cut_callback_time'] + control['total_heuristic_callback_time']
    control['total_solver_time'] = control['total_run_time'] - control['total_callback_time']
    control['total_data_time'] = control['total_cut_time'] + control['total_polish_time'] + control['total_round_time'] + control['total_round_then_polish_time']

    # General Output

    model_info = {
        'solution': control['incumbent'],
        'objective_value': get_objval(control['incumbent']) if control['found_solution'] else float('inf'),
        'loss_value': compute_loss(control['incumbent']) if control['found_solution'] else float('inf'),
        'optimality_gap': control['relative_gap'] if control['found_solution'] else float('inf'),
        #
        'run_time': control['total_run_time'],
        'solver_time': control['total_solver_time'],
        'callback_time': control['total_callback_time'],
        'data_time': control['total_data_time'],
        'nodes_processed': control['nodes_processed'],
        #
        'c0_value': c0_value,
        'w_pos': settings['w_pos'],
        }

    model_info.update(constraints)

    # MIP object
    mip_info = {
        'risk_slim_mip': risk_slim_mip,
        'risk_slim_idx': indices
        }

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

    def initialize(self, indices, control, settings, compute_loss_cut, get_alpha, get_L0_penalty_from_alpha, initial_cuts = None, cut_queue = None, polish_queue = None):

        self.settings = settings  #store pointer to shared settings so that settings can be turned on/off during B&B
        self.control = control  # dict containing information for flow
        self.rho_idx = indices['rho']
        self.cut_idx = indices['loss'] + indices['rho']
        self.alpha_idx = indices['alpha']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']

        # todo (validate initial cutting planes)
        self.initial_cuts = initial_cuts
        self.compute_loss_cut = compute_loss_cut
        self.get_alpha = get_alpha
        self.get_L0_penalty_from_alpha = get_L0_penalty_from_alpha

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
            self.add(constraint = self.loss_cut_constraint, sense = "G", rhs = bounds['loss_min'], use = self.bound_cut_purge_flag)
            self.control['bounds']['loss_min'] = bounds['loss_min']
            self.control['n_bound_updates_loss_min'] += 1

        if bounds['objval_min'] > self.control['bounds']['objval_min']:
            self.add(constraint = self.objval_cut_constraint, sense = "G", rhs = bounds['objval_min'], use = self.bound_cut_purge_flag)
            self.control['bounds']['objval_min'] = bounds['objval_min']
            self.control['n_bound_updates_objval_min'] += 1

        if bounds['L0_max'] < self.control['bounds']['L0_max']:
            self.add(constraint = self.L0_cut_constraint, sense="L", rhs = bounds['L0_max'], use = self.bound_cut_purge_flag)
            self.control['bounds']['L0_max'] = bounds['L0_max']
            self.control['n_bound_updates_L0_max'] += 1

        if bounds['loss_max'] < self.control['bounds']['loss_max']:
            self.add(constraint = self.loss_cut_constraint, sense="L", rhs = bounds['loss_max'], use = self.bound_cut_purge_flag)
            self.control['bounds']['loss_max'] = bounds['loss_max']
            self.control['n_bound_updates_loss_max'] += 1

        if bounds['objval_max'] < self.control['bounds']['objval_max']:
            self.add(constraint = self.objval_cut_constraint, sense="L", rhs = bounds['objval_max'], use = self.bound_cut_purge_flag)
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
                self.add(constraint = self.initial_cuts['coefs'][i],
                         sense = "G",
                         rhs = self.initial_cuts['lhs'][i],
                         use = self.loss_cut_purge_flag)
            self.initial_cuts = None

        # get integer feasible solution
        rho = np.array(self.get_values(self.rho_idx))
        alpha = np.array(self.get_values(self.alpha_idx))

        # check that CPLEX is actually integer. if not, then recast as int
        if ~is_integer(rho):
            rho = cast_to_integer(rho)
            alpha = self.get_alpha(rho)

        # add cutting plane at integer feasible solution
        cut_start_time = time.time()
        loss_value, loss_slope = self.compute_loss_cut(rho)
        cut_lhs = float(loss_value - loss_slope.dot(rho))
        cut_constraint = [self.cut_idx, [1.0] + (-loss_slope).tolist()]
        self.add(constraint=cut_constraint, sense="G", rhs=cut_lhs, use=self.loss_cut_purge_flag)
        cut_time = time.time() - cut_start_time
        cuts_added = 1

        # if solution updates incumbent, then add solution to queue for polishing
        current_upperbound = float(loss_value + self.get_L0_penalty_from_alpha(alpha))
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
                    [loss_value, loss_slope] = self.compute_loss_cut(cut_rho)
                    self.add(constraint = [self.cut_idx, [1.0] + (-loss_slope).tolist()],
                             sense = "G",
                             rhs = float(loss_value - loss_slope.dot(cut_rho)),
                             use = self.loss_cut_purge_flag)
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

    def initialize(self, indices, control, settings, cut_queue, polish_queue, get_objval, get_L0_norm, is_feasible, dcd_polishing, sequential_rounding):

        #todo: add basic assertions to make sure that nothing weird is going on
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

        assert callable(get_objval)
        assert callable(get_L0_norm)
        assert callable(is_feasible)
        assert callable(dcd_polishing)
        assert callable(sequential_rounding)

        self.get_objval = get_objval
        self.get_L0_norm = get_L0_norm
        self.is_feasible = is_feasible
        self.sequential_rounding = sequential_rounding
        self.dcd_polishing = dcd_polishing
        return


    def update_heuristic_flags(self):

        # keep on rounding?
        keep_rounding = (self.settings['rounding_start_cuts'] <= self.control['n_cuts'] <= self.settings['rounding_stop_cuts'] and
                         self.settings['rounding_stop_gap'] <= self.control['relative_gap'] <= self.settings['rounding_start_gap'])

        # keep on polishing?
        keep_polishing = (self.settings['polishing_start_cuts'] <= self.control['n_cuts'] <= self.settings['polishing_stop_cuts'] and
                          self.settings['polishing_stop_gap'] <= self.control['relative_gap'] <= self.settings['polishing_start_gap'])

        self.polish_flag &= keep_polishing
        self.round_flag &= keep_rounding
        self.polish_rounded_solutions &= self.round_flag
        return


    def get_L0_range_of_rounded_solution(self, rho):
        abs_rho = abs(rho)
        rounded_L0_min = self.get_L0_norm(np.floor(abs_rho))
        rounded_L0_max = self.get_L0_norm(np.ceil(abs_rho))
        return rounded_L0_min, rounded_L0_max

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
                    self.control['upperbound'] = self.get_objval(cplex_incumbent)

        #decide whether to keep on rounding/polishing
        self.update_heuristic_flags()
        #variables to store best objective value / solution from heuristics
        best_objval = float('inf')
        best_solution = None

        # run sequential rounding if lower bound was updated since the last call
        if self.round_flag and lowerbound_update:

            rho_cts = np.array(self.get_values(self.rho_idx))
            min_l0_norm, max_l0_norm = self.get_L0_range_of_rounded_solution(rho_cts)
            rounded_solution_is_feasible = (min_l0_norm < self.control['bounds']['L0_max'] and max_l0_norm > self.control['bounds']['L0_min'])

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

                    if self.is_feasible(rounded_solution, L0_min=  self.control['bounds']['L0_min'], L0_max = self.control['bounds']['L0_max']):
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

                            if self.is_feasible(polished_solution, L0_min=self.control['bounds']['L0_min'], L0_max=self.control['bounds']['L0_max']):
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
                    if self.is_feasible(polished_solution, L0_min = self.control['bounds']['L0_min'], L0_max = self.control['bounds']['L0_max']):
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
