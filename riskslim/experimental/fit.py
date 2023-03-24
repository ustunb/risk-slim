import time
import numpy as np
import warnings

from scipy.special import expit
from cplex.exceptions import CplexError
from cplex.callbacks import HeuristicCallback, LazyConstraintCallback

from riskslim.utils import cast_to_integer, check_data, is_integer, validate_settings, print_log
from riskslim.defaults import DEFAULT_LCPA_SETTINGS
from riskslim.coefficient_set import CoefficientSet, get_score_bounds
from riskslim.mip import add_mip_starts, convert_to_risk_slim_cplex_solution, create_risk_slim, set_cplex_mip_parameters
from riskslim.solution_pool import SolutionPool, FastSolutionPool
from riskslim.experimental.speedups.heuristics import discrete_descent, sequential_rounding
from riskslim.experimental.speedups.bound_tightening import Bounds, chained_updates
from riskslim.experimental.speedups.warmstart import run_standard_cpa, round_solution_pool,sequential_round_solution_pool, discrete_descent_solution_pool

class RiskSLIMFitter(object):

    _default_print_flag = False

    def __init__(self, data, constraints, **kwargs):

        # empty fields
        self._fitted = False
        self._has_warmstart = False
        self.initial_cuts = None

        # settings
        self.settings, self.cplex_settings, self.warmstart_settings = self._parse_settings(kwargs, defaults = DEFAULT_LCPA_SETTINGS)

        # data
        check_data(data)
        self._Z = data['X'] * data['Y']
        self._integer_data = np.all(self._Z == np.require(self._Z, dtype = np.int_))
        self.n_variables = self._Z.shape[1]

        # coefficient constraints
        self.coef_set = constraints['coef_set']
        self.rho_lb = np.array(constraints['coef_set'].lb)
        self.rho_ub = np.array(constraints['coef_set'].ub)

        # regularization parameter
        c0_value = float(self.settings['c0_value'])
        assert c0_value > 0.0, 'C0 should be positive'
        self.c0_value = c0_value

        # vectorized regularization parameters
        self.L0_reg_ind = np.isnan(self.coef_set.c0)
        self.C_0 = np.array(self.coef_set.c0)
        self.C_0[self.L0_reg_ind] = c0_value
        self.C_0_nnz = self.C_0[self.L0_reg_ind]

        # model size constraints
        self.L0_max = np.minimum(constraints['L0_max'], np.sum(self.L0_reg_ind))
        self.L0_min = np.maximum(constraints['L0_min'], 0)

        # function handles
        self._init_loss_computation(data = data, loss_computation = self.settings['loss_computation'])
        self.get_L0_norm = lambda rho: np.count_nonzero(rho[self.L0_reg_ind])
        self.get_L0_penalty = lambda rho: np.sum(self.C_0_nnz * (rho[self.L0_reg_ind] != 0.0))
        self.get_alpha = lambda rho: np.array(abs(rho[self.L0_reg_ind]) > 0.0, dtype = np.float_)
        self.get_L0_penalty_from_alpha = lambda alpha: np.sum(self.C_0_nnz * alpha)
        self.get_objval = lambda rho: self.compute_loss(rho) + np.sum(self.C_0_nnz * (rho[self.L0_reg_ind] != 0.0))

        # bounds
        self.bounds = Bounds(L0_min = self.L0_min, L0_max = self.L0_max)
        self.bounds.loss_min, self.bounds.loss_max = self._init_loss_bounds()

        # initialize
        self.pool = SolutionPool(self.n_variables)

        # check if trivial solution is feasible, if so add it to the pool and update bounds
        trivial_solution = np.zeros(self.n_variables)
        if self.is_feasible(trivial_solution):
            trivial_objval = self.compute_loss(trivial_solution)
            if self.settings['initial_bound_updates']:
                self.bounds.objval_max = min(self.bounds.objval_max, trivial_objval)
                self.bounds.loss_max = min(self.bounds.loss_max, trivial_objval)
                self.bounds = chained_updates(self.bounds, self.C_0_nnz)
            self.pool.add(objvals = trivial_objval, solutions = trivial_solution)

        # setup mip
        mip_settings = {
            'C_0': c0_value,
            'coef_set': constraints['coef_set'],
            'tight_formulation': self.settings['tight_formulation'],
            'drop_variables': self.settings['drop_variables'],
            'include_auxillary_variable_for_L0_norm': self.settings['include_auxillary_variable_for_L0_norm'],
            'include_auxillary_variable_for_objval': self.settings['include_auxillary_variable_for_objval']
            }

        mip_settings.update(self.bounds._asdict())
        self.mip_settings = mip_settings

        # setup risk slim mip
        self._mip, self._mip_indices = create_risk_slim(coef_set = constraints['coef_set'], input = self.mip_settings)
        self._mip_indices.update({
            'C_0_nnz': self.C_0_nnz,
            'L0_reg_ind': self.L0_reg_ind,
            })

        self.stats = {
            'incumbent': np.repeat(np.nan, self.n_variables),
            'upperbound': float('inf'),
            'bounds': self.bounds,
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
            'heuristic_callback_times_called': 0,
            'total_cut_callback_time': 0.00,
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

    def warmstart(self, **kwargs):
        """
        Run an initialization routine to speed up RiskSLIM
        """

        settings = validate_settings(kwargs, defaults = self.warmstart_settings)
        settings['type'] = 'cvx'

        # construct LP relaxation
        lp_settings = dict(self.mip_settings)
        lp_settings['relax_integer_variables'] = True
        cpx, cpx_indices = create_risk_slim(coef_set = self.coef_set, input = lp_settings)
        cpx = set_cplex_mip_parameters(cpx, self.cplex_settings, display_cplex_progress = settings['display_cplex_progress'])

        # solve RiskSLIMFitter LP using standard CPA
        stats, cuts, pool = run_standard_cpa(cpx = cpx, cpx_indices = cpx_indices, compute_loss = self.compute_loss_real, compute_loss_cut = self.compute_loss_cut_real, settings = settings)

        # update cuts
        print_log('warmstart CPA produced %d cuts' % len(cuts['coefs']))
        self.initial_cuts = cuts

        # update bounds
        bounds = chained_updates(self.bounds, self.C_0_nnz, new_objval_at_relaxation = stats['lowerbound'])

        constraints = {
            'L0_min': self.L0_min,
            'L0_max': self.L0_max,
            'coef_set': self.coef_set,
            }

        def rounded_model_size_is_ok(rho):
            zero_idx_rho_ceil = np.equal(np.ceil(rho), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho), 0)
            cannot_round_to_zero = np.logical_not(np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor))
            rounded_rho_L0_min = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            rounded_rho_L0_max = np.count_nonzero(rho[self.L0_reg_ind])
            return rounded_rho_L0_min >= self.L0_min >= 0 and rounded_rho_L0_max <= self.L0_max

        pool = pool.remove_infeasible(rounded_model_size_is_ok).distinct().sort()

        if len(pool) == 0:
            print_log('all CPA solutions are infeasible')

        pool = SolutionPool(pool.P)

        # round CPA solutions
        if settings['use_rounding'] and len(pool) > 0:

            print_log('running naive rounding on %d solutions' % len(pool))
            print_log('best objective value: %1.4f' % np.min(pool.objvals))
            rnd_pool, _, _ = round_solution_pool(pool, constraints, max_runtime = settings['rounding_max_runtime'], max_solutions = settings['rounding_max_solutions'])
            rnd_pool = rnd_pool.compute_objvals(self.get_objval).remove_infeasible(self.is_feasible)
            print_log('rounding produced %d integer solutions' % len(rnd_pool))

            if len(rnd_pool) > 0:
                pool.append(rnd_pool)
                print_log('best objective value is %1.4f' % np.min(rnd_pool.objvals))

        # sequentially round CPA solutions
        if settings['use_sequential_rounding'] and len(pool) > 0:

            print_log('running sequential rounding on %d solutions' % len(pool))
            print_log('best objective value: %1.4f' % np.min(pool.objvals))

            sqrnd_pool, _, _ = sequential_round_solution_pool(
                    pool = pool,
                    Z = self._Z,
                    C_0 = self.C_0,
                    compute_loss_from_scores_real = self.compute_loss_from_scores_real,
                    get_L0_penalty = self.get_L0_penalty,
                    max_runtime = settings['sequential_rounding_max_runtime'],
                    max_solutions = settings['sequential_rounding_max_solutions'],
                    objval_cutoff = self.bounds.objval_max)

            sqrnd_pool = sqrnd_pool.remove_infeasible(self.is_feasible)
            print_log('sequential rounding produced %d integer solutions' % len(sqrnd_pool))

            if len(sqrnd_pool) > 0:
                pool = pool.append(sqrnd_pool)
                print_log('best objective value: %1.4f' % np.min(pool.objvals))

        # polish rounded solutions
        if settings['polishing_after'] and len(pool) > 0:
            print_log('polishing %d solutions' % len(pool))
            print_log('best objective value: %1.4f' % np.min(pool.objvals))
            dcd_pool, _, _ = discrete_descent_solution_pool(pool = pool,
                                                            Z = self._Z,
                                                            C_0 = self.C_0,
                                                            constraints = constraints,
                                                            compute_loss_from_scores = self.compute_loss_from_scores,
                                                            get_L0_penalty = self.get_L0_penalty,
                                                            max_runtime = settings['polishing_max_runtime'],
                                                            max_solutions = settings['polishing_max_solutions'])

            dcd_pool = dcd_pool.remove_infeasible(self.is_feasible)
            if len(dcd_pool) > 0:
                print_log('polishing produced %d integer solutions' % len(dcd_pool))
                pool.append(dcd_pool)

        # remove solutions that are not feasible, not integer
        if len(pool) > 0:
            pool = pool.remove_nonintegral().distinct().sort()

        # update upper and lower bounds
        print_log('initialization produced %1.0f feasible solutions' % len(pool))

        if len(pool) > 0:
            bounds = chained_updates(bounds, self.C_0_nnz, new_objval_at_feasible = np.min(pool.objvals))
            print_log('best objective value: %1.4f' % np.min(pool.objvals))

        self.pool.append(pool)

        # update bounds
        self.bounds = bounds.__copy__()
        self._has_warmstart = True

    def fit(self, **kwargs):

        if len(kwargs) > 0:
            current_settings = dict(self.settings)
            current_settings.update(self.cplex_settings)
            current_settings.update(self.warmstart_settings)
            new_settings, new_cplex_settings, new_warmstart_settings = self._parse_settings(kwargs, defaults = current_settings)
            self.settings.update(new_settings)
            self.new_cplex_settings.update(new_cplex_settings)
            self.new_warmstart_settings.update(new_warmstart_settings)

        # run warmstart procedure if it has not been run yet
        if self.settings['initialization_flag'] and not self.has_warmstart:
            self.warmstart(**self.warmstart_settings)

        # set cplex parameters and runtime
        cpx = self._mip
        cpx = set_cplex_mip_parameters(cpx, self.cplex_settings, display_cplex_progress = self.settings['display_cplex_progress'])
        cpx.parameters.timelimit.set(self.settings['max_runtime'])

        # initialize solution queues
        self.cut_queue = FastSolutionPool(self.n_variables)
        self.polish_queue = FastSolutionPool(self.n_variables)

        if not self.fitted:
            self._init_callbacks()

        # initialize solution pool
        if len(self.pool) > 0:
            # initialize using the polish_queue when possible to avoid bugs with the CPLEX MIPStart interface
            if self.settings['polish_flag']:
                self.polish_queue.add(self.pool.objvals[0], self.pool.solutions[0])
            else:
                cpx = add_mip_starts(cpx, self.mip_indices, self.pool, mip_start_effort_level = cpx.MIP_starts.effort_level.repair)

            if self.settings['add_cuts_at_heuristic_solutions'] and len(self.pool) > 1:
                self.cut_queue.add(self.pool.objvals[1:], self.pool.solutions[1:])

        # solve riskslim optimization problem
        start_time = time.time()
        cpx.solve()
        self.stats['total_run_time'] = time.time() - start_time
        self._fitted = True

    #### flags ####
    @property
    def print_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._print_flag

    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = RiskSLIMFitter._default_print_flag
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise ValueError('print_flag must be boolean or None')
        self._toggle_mip_display(cpx = self.mip, flag = self._print_flag)

    @property
    def fitted(self):
        return self._fitted

    @property
    def has_warmstart(self):
        """
        Returns True if this instance has already been initialized
        """
        return self._has_warmstart

    #### properties ####
    @property
    def mip(self):
        return self._mip

    @property
    def mip_indices(self):
        return self._mip_indices

    @property
    def solution(self):
        """
        :return: handle to CPLEX solution
        """
        # todo add wrapper if solution does not exist
        return self.mip.solution

    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        #get_mip_stats(self.mip)
        # record mip solution statistics
        s = self.solution

        try:
            self.stats['incumbent'] = np.array(s.get_values(self.mip_indices['rho']))
            self.stats['upperbound'] = s.get_objective_value()
            self.stats['lowerbound'] = s.MIP.get_best_objective()
            self.stats['relative_gap'] = s.MIP.get_mip_relative_gap()
            self.stats['found_solution'] = True
        except CplexError:
            self.stats['found_solution'] = False

        self.stats['cplex_status'] = s.get_status_string()
        self.stats['total_callback_time'] = self.stats['total_cut_callback_time'] + self.stats['total_heuristic_callback_time']
        self.stats['total_solver_time'] = self.stats['total_run_time'] - self.stats['total_callback_time']
        self.stats['total_data_time'] = self.stats['total_cut_time'] + self.stats['total_polish_time'] + self.stats['total_round_time'] + self.stats['total_round_then_polish_time']

        # Output for Model
        info = {
            'c0_value': self.c0_value,
            #
            'solution': self.stats['incumbent'],
            'objective_value': self.get_objval(self.stats['incumbent']) if self.stats['found_solution'] else float('inf'),
            'loss_value': self.compute_loss(self.stats['incumbent']) if self.stats['found_solution'] else float('inf'),
            'optimality_gap': self.stats['relative_gap'] if self.stats['found_solution'] else float('inf'),
            #
            'run_time': self.stats['total_run_time'],
            'solver_time': self.stats['total_solver_time'],
            'callback_time': self.stats['total_callback_time'],
            'data_time': self.stats['total_data_time'],
            'nodes_processed': self.stats['nodes_processed'],
            }

        return info

    @property
    def Z(self):
        return self._Z

    @property
    def loss_computation(self):
        return self._loss_computation

    @property
    def coefficients(self):
        """
        :return: coefficients of the linear classifier
        """
        s = self.solution
        if s.is_primal_feasible():
            coefs = np.array(s.get_values(self.mip_indices['rho']))
        else:
            coefs = np.repeat(np.nan, self.n_variables)
        return coefs

    #### classifier API ####
    def predict(self, X):
        assert self.fitted
        return np.sign(X.dot(self.coefficients))

    def predict_proba(self, X):
        assert self.fitted
        return expit(X.dot(self.coefficients))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))

    # helper functions
    def is_feasible(self, rho):
        return np.all(self.rho_ub >= rho) and np.all(self.rho_lb <= rho) and (self.L0_min <= np.count_nonzero(rho[self.L0_reg_ind]) <= self.L0_max)

    ### initialization ####
    def _parse_settings(self, settings, defaults):
        """
        Parameters
        ----------
        settings
        defaults

        Returns
        -------

        """

        settings = validate_settings(settings, defaults)
        warmstart_settings = {k.lstrip('init_'): settings[k] for k in settings if k.startswith('init_')}
        cplex_settings = {k.lstrip('cplex_'): settings[k] for k in settings if k.startswith('cplex_')}
        lcpa_settings = {k: settings[k] for k in settings if settings if not k.startswith(('init_', 'cplex_'))}
        return lcpa_settings, cplex_settings, warmstart_settings

    def _init_loss_computation(self, data, loss_computation = 'normal', w_pos = 1.0):
        """

        Parameters
        ----------
        data
        coef_set
        L0_max
        loss_computation
        w_pos

        Returns
        -------

        """
        # todo check if fast/lookup loss is installed
        assert loss_computation in ['normal', 'weighted', 'fast', 'lookup']

        use_weighted = False
        use_lookup_table = isinstance(self.coef_set, CoefficientSet) and self._integer_data

        if 'sample_weights' in data:
            sample_weights = self._init_training_weights(Y = data['Y'], sample_weights = data['sample_weights'], w_pos = w_pos)
            use_weighted = not np.all(np.equal(sample_weights, 1.0))

        if use_weighted:
            final_loss_computation = 'weighted'
        elif use_lookup_table:
            final_loss_computation = 'lookup'
        else:
            final_loss_computation = 'fast'

        if final_loss_computation != loss_computation:
            warnings.warn("switching loss computation from %s to %s" % (loss_computation, final_loss_computation))

        self._loss_computation = final_loss_computation

        if final_loss_computation == 'normal':
            import riskslim.loss_functions.log_loss as lf
            self._Z = np.require(self._Z, requirements = ['C'])
            self.compute_loss = lambda rho: lf.log_loss_value(self._Z, rho)
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self._Z, rho)
            self.compute_loss_from_scores = lambda scores: lf.log_loss_value_from_scores(scores)

        elif final_loss_computation == 'fast':

            import riskslim.loss_functions.fast_log_loss as lf
            self._Z = np.require(self._Z, requirements = ['F'])
            self.compute_loss = lambda rho: lf.log_loss_value(self._Z, rho)
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self._Z, rho)
            self.compute_loss_from_scores = lambda scores: lf.log_loss_value_from_scores(scores)

        elif final_loss_computation == 'weighted':

            import riskslim.loss_functions.log_loss_weighted as lf
            self._Z = np.require(self._Z, requirements = ['C'])
            total_sample_weights = np.sum(sample_weights)
            self.compute_loss = lambda rho: lf.log_loss_value(self._Z, sample_weights, total_sample_weights, rho)
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self._Z, sample_weights, total_sample_weights, rho)
            self.compute_loss_from_scores = lambda scores: lf.log_loss_value_from_scores(sample_weights, total_sample_weights, scores)

        elif final_loss_computation == 'lookup':

            import riskslim.loss_functions.lookup_log_loss as lf
            self._Z = np.require(self._Z, requirements = ['F'], dtype = np.float64)

            s_min, s_max = get_score_bounds(Z_min = np.min(self._Z, axis = 0),
                                            Z_max = np.max(self._Z, axis = 0),
                                            rho_lb = self.coef_set.lb,
                                            rho_ub = self.coef_set.ub,
                                            L0_reg_ind = np.array(self.coef_set.c0) == 0.0,
                                            L0_max = self.L0_max)


            print_log("%d rows in lookup table" % (s_max - s_min + 1))
            loss_value_tbl, prob_value_tbl, tbl_offset = lf.get_loss_value_and_prob_tables(s_min, s_max)
            self.compute_loss = lambda rho: lf.log_loss_value(self._Z, rho, loss_value_tbl,tbl_offset)
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self._Z, rho, loss_value_tbl, prob_value_tbl, tbl_offset)
            self.compute_loss_from_scores = lambda scores: lf.log_loss_value_from_scores(scores, loss_value_tbl, tbl_offset)

        # real loss functions
        if final_loss_computation == 'lookup':

            import riskslim.loss_functions.fast_log_loss as lfr
            self.compute_loss_real = lambda rho: lfr.log_loss_value(self._Z, rho)
            self.compute_loss_cut_real = lambda rho: lfr.log_loss_value_and_slope(self._Z, rho)
            self.compute_loss_from_scores_real = lambda scores: lfr.log_loss_value_from_scores(scores)

        else:
            self.compute_loss_real = self.compute_loss
            self.compute_loss_real = self.compute_loss_cut
            self.compute_loss_real = self.compute_loss_from_scores

    def _init_training_weights(self, Y, sample_weights = None, w_pos = 1.0, w_neg = 1.0, w_total_target = 2.0):

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
        assert np.isfinite(w_pos), 'w_pos must be finite'
        assert np.isfinite(w_neg), 'w_neg must be finite'
        assert w_pos > 0.0, 'w_pos must be strictly positive'
        assert w_neg > 0.0, 'w_neg must be strictly positive'
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
            # todo: throw warning if any training weights = 0
            # todo: throw warning if there are no effective positive/negative points in Y

        # normalization
        training_weights = N * (training_weights / sum(training_weights))
        training_weights[pos_ind] *= w_pos
        training_weights[~pos_ind] *= w_neg

        return training_weights

    def _init_callbacks(self):
        """
        Initializes callback functions
        """

        loss_cb = self.mip.register_callback(LossCallback)
        loss_cb.initialize(indices = self.mip_indices,
                           stats = self.stats,
                           settings = self.settings,
                           compute_loss_cut = self.compute_loss_cut,
                           get_alpha = self.get_alpha,
                           get_L0_penalty_from_alpha = self.get_L0_penalty_from_alpha,
                           initial_cuts = self.initial_cuts,
                           cut_queue = self.cut_queue,
                           polish_queue = self.polish_queue)

        self.loss_callback = loss_cb

        # add heuristic callback if rounding or polishing
        if self.settings['round_flag'] or self.settings['polish_flag']:

            heuristic_cb = self.mip.register_callback(PolishAndRoundCallback)
            active_set_flag = self.L0_max <= self.n_variables
            polisher = lambda rho: discrete_descent(rho,
                                                    self.Z,
                                                    self.C_0,
                                                    self.rho_ub,
                                                    self.rho_lb,
                                                    self.get_L0_penalty,
                                                    self.compute_loss_from_scores,
                                                    active_set_flag)

            rounder = lambda rho, cutoff: sequential_rounding(rho,
                                                              self.Z,
                                                              self.C_0,
                                                              self.compute_loss_from_scores_real,
                                                              self.get_L0_penalty,
                                                              cutoff)

            heuristic_cb.initialize(indices = self.mip_indices,
                                    control = self.stats,
                                    settings = self.settings,
                                    cut_queue = self.cut_queue,
                                    polish_queue = self.polish_queue,
                                    get_objval = self.get_objval,
                                    get_L0_norm = self.get_L0_norm,
                                    is_feasible = self.is_feasible,
                                    polishing_handle = polisher,
                                    rounding_handle = rounder)

            self.heuristic_callback = heuristic_cb

    def _init_loss_bounds(self):
        # min value of loss = log(1+exp(-score)) occurs at max score for each point
        # max value of loss = loss(1+exp(-score)) occurs at min score for each point

        # get maximum number of regularized coefficients
        num_max_reg_coefs = self.L0_max

        # calculate the smallest and largest score that can be attained by each point
        scores_at_lb = self._Z * self.rho_lb
        scores_at_ub = self._Z * self.rho_ub
        max_scores_matrix = np.maximum(scores_at_ub, scores_at_lb)
        min_scores_matrix = np.minimum(scores_at_ub, scores_at_lb)
        assert (np.all(max_scores_matrix >= min_scores_matrix))

        # for each example, compute max sum of scores from top reg coefficients
        max_scores_reg = max_scores_matrix[:, self.L0_reg_ind]
        max_scores_reg = -np.sort(-max_scores_reg, axis = 1)
        max_scores_reg = max_scores_reg[:, 0:num_max_reg_coefs]
        max_score_reg = np.sum(max_scores_reg, axis = 1)

        # for each example, compute max sum of scores from no reg coefficients
        max_scores_no_reg = max_scores_matrix[:, ~self.L0_reg_ind]
        max_score_no_reg = np.sum(max_scores_no_reg, axis = 1)

        # max score for each example
        max_score = max_score_reg + max_score_no_reg

        # for each example, compute min sum of scores from top reg coefficients
        min_scores_reg = min_scores_matrix[:, self.L0_reg_ind]
        min_scores_reg = np.sort(min_scores_reg, axis = 1)
        min_scores_reg = min_scores_reg[:, 0:num_max_reg_coefs]
        min_score_reg = np.sum(min_scores_reg, axis = 1)

        # for each example, compute min sum of scores from no reg coefficients
        min_scores_no_reg = min_scores_matrix[:, ~self.L0_reg_ind]
        min_score_no_reg = np.sum(min_scores_no_reg, axis = 1)

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


class LossCallback(LazyConstraintCallback):
    """
    Call initialize() before using this callback.

    LossCallback is called when CPLEX finds an integer feasible solution. By default, it will add a cut at this
    solution to improve the cutting-plane approximation of the loss function. The cut is added as a 'lazy' constraint
    into the surrogate LP so that it is evaluated only when necessary.

    Optional Functionality:

    - add an initial set of cutting planes found by warm starting
      requires initial_cuts

    - pass integer feasible solutions to 'polish' queue so that they can be polished with DCD in the PolishAndRoundCallback
      requires settings['polish_flag'] = True

    - adds cuts at integer feasible solutions found by the PolishAndRoundCallback
      requires settings['add_cuts_at_heuristic_solutions'] = True

    - reduces overall search region by adding constraints on objval_max, l0_max, loss_min, loss_max
      requires settings['chained_updates_flag'] = True
    """

    def initialize(self, indices, stats, settings, compute_loss_cut, get_alpha, get_L0_penalty_from_alpha, initial_cuts = None, cut_queue = None, polish_queue = None):

        assert isinstance(indices, dict)
        assert isinstance(stats, dict)
        assert isinstance(settings, dict)
        assert callable(compute_loss_cut)
        assert callable(get_alpha)
        assert callable(get_L0_penalty_from_alpha)


        self.settings = settings  #store pointer to shared settings so that settings can be turned on/off during B&B
        self.stats = stats  # dict containing information for flow

        # todo (validate initial cutting planes)
        self.initial_cuts = initial_cuts

        # indices
        self.rho_idx = indices['rho']
        self.cut_idx = indices['loss'] + indices['rho']
        self.alpha_idx = indices['alpha']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']
        self.compute_loss_cut = compute_loss_cut
        self.get_alpha = get_alpha
        self.get_L0_penalty_from_alpha = get_L0_penalty_from_alpha

        # cplex can drop cutting planes that are not used. by default, we force CPLEX to use all cutting planes.
        self.loss_cut_purge_flag = self.use_constraint.purge if self.settings['purge_loss_cuts'] else self.use_constraint.force

        # setup pointer to cut_queue to receive cuts from PolishAndRoundCallback
        if self.settings['add_cuts_at_heuristic_solutions']:
            if cut_queue is None:
                self.cut_queue = FastSolutionPool(len(self.rho_idx))
            else:
                assert isinstance(cut_queue, FastSolutionPool)
                self.cut_queue = cut_queue

        # setup pointer to polish_queue to send integer solutions to PolishAndRoundCallback
        if self.settings['polish_flag']:
            if polish_queue is None:
                self.polish_queue = FastSolutionPool(len(self.rho_idx))
            else:
                assert isinstance(polish_queue, FastSolutionPool)
                self.polish_queue = polish_queue

        # setup indices for chained_updates
        if self.settings['chained_updates_flag']:
            self.loss_cut_constraint = [indices['loss'], [1.0]]
            self.L0_cut_constraint = [[indices['L0_norm']], [1.0]]
            self.objval_cut_constraint = [[indices['objval']], [1.0]]
            self.bound_cut_purge_flag = self.use_constraint.purge if self.settings['purge_loss_cuts'] else self.use_constraint.force

        return

    def add_loss_cut(self, rho):

        loss_value, loss_slope = self.compute_loss_cut(rho)
        self.add(constraint = [self.cut_idx, [1.0] + (-loss_slope).tolist()],
                 sense = "G",
                 rhs = float(loss_value - loss_slope.dot(rho)),
                 use = self.loss_cut_purge_flag)

        return loss_value

    def update_bounds(self):
        """

        Returns
        -------

        """

        bounds = chained_updates(bounds = self.stats['bounds'],
                                 C_0_nnz = self.C_0_nnz,
                                 new_objval_at_relaxation = self.stats['lowerbound'],
                                 new_objval_at_feasible = self.stats['upperbound'])

        #add cuts if bounds need to be tighter
        if bounds.loss_min > self.stats['bounds'].loss_min:
            self.add(constraint = self.loss_cut_constraint, sense = "G", rhs = bounds.loss_min, use = self.bound_cut_purge_flag)
            self.stats['bounds'].loss_min = bounds.loss_min
            self.stats['n_bound_updates_loss_min'] += 1

        if bounds.objval_min > self.stats['bounds'].objval_min:
            self.add(constraint = self.objval_cut_constraint, sense = "G", rhs = bounds.objval_min, use = self.bound_cut_purge_flag)
            self.stats['bounds'].objval_min = bounds.objval_min
            self.stats['n_bound_updates_objval_min'] += 1

        if bounds.L0_max < self.stats['bounds'].L0_max:
            self.add(constraint = self.L0_cut_constraint, sense="L", rhs = bounds.L0_max, use = self.bound_cut_purge_flag)
            self.stats['bounds'].L0_max = bounds.L0_max
            self.stats['n_bound_updates_L0_max'] += 1

        if bounds.loss_max < self.stats['bounds'].loss_max:
            self.add(constraint = self.loss_cut_constraint, sense="L", rhs = bounds.loss_max, use = self.bound_cut_purge_flag)
            self.stats['bounds'].loss_max = bounds.loss_max
            self.stats['n_bound_updates_loss_max'] += 1

        if bounds.objval_max < self.stats['bounds'].objval_max:
            self.add(constraint = self.objval_cut_constraint, sense="L", rhs = bounds.objval_max, use = self.bound_cut_purge_flag)
            self.stats['bounds'].objval_max = bounds.objval_max
            self.stats['n_bound_updates_objval_max'] += 1

        return

    def __call__(self):

        #print_log('in cut callback')
        callback_start_time = time.time()

        #record entry metrics
        self.stats['cut_callback_times_called'] += 1
        self.stats['lowerbound'] = self.get_best_objective_value()
        self.stats['relative_gap'] = self.get_MIP_relative_gap()
        self.stats['nodes_processed'] = self.get_num_nodes()
        self.stats['nodes_remaining'] = self.get_num_remaining_nodes()

        # add initial cuts first time the callback is used
        if self.initial_cuts is not None:
            print_log('adding %1.0f initial cuts' % len(self.initial_cuts['lhs']))
            for cut, lhs in zip(self.initial_cuts['coefs'], self.initial_cuts['lhs']):
                self.add(constraint = cut, sense = "G", rhs = lhs, use = self.loss_cut_purge_flag)
            self.initial_cuts = None

        # get integer feasible solution
        rho = np.array(self.get_values(self.rho_idx))
        alpha = np.array(self.get_values(self.alpha_idx))

        # check that CPLEX is actually integer. if not, then recast as int
        if not is_integer(rho):
            rho = cast_to_integer(rho)
            alpha = self.get_alpha(rho)

        # add cutting plane at integer feasible solution
        cut_start_time = time.time()
        loss_value = self.add_loss_cut(rho)
        cut_time = time.time() - cut_start_time
        cuts_added = 1

        # if solution updates incumbent, then add solution to queue for polishing
        current_upperbound = float(loss_value + self.get_L0_penalty_from_alpha(alpha))
        incumbent_update = current_upperbound < self.stats['upperbound']

        if incumbent_update:
            self.stats['incumbent'] = rho
            self.stats['upperbound'] = current_upperbound
            self.stats['n_incumbent_updates'] += 1

        if self.settings['polish_flag']:
            polishing_cutoff = self.stats['upperbound'] * (1.0 + self.settings['polishing_tolerance'])
            if current_upperbound < polishing_cutoff:
                self.polish_queue.add(current_upperbound, rho)

        # add cutting planes at other integer feasible solutions in cut_queue
        if self.settings['add_cuts_at_heuristic_solutions']:
            if len(self.cut_queue) > 0:
                self.cut_queue.filter_sort_unique()
                cut_start_time = time.time()
                for cut_rho in self.cut_queue.solutions:
                    self.add_loss_cut(cut_rho)
                cut_time += time.time() - cut_start_time
                cuts_added += len(self.cut_queue)
                self.cut_queue.clear()

        # update bounds
        if self.settings['chained_updates_flag']:
            if (self.stats['lowerbound'] > self.stats['bounds'].objval_min) or (self.stats['upperbound'] < self.stats['bounds'].objval_max):
                self.stats['n_update_bounds_calls'] += 1
                self.update_bounds()

        # record metrics at end
        self.stats['n_cuts'] += cuts_added
        self.stats['total_cut_time'] += cut_time
        self.stats['total_cut_callback_time'] += time.time() - callback_start_time

        #print_log('left cut callback')
        return


class PolishAndRoundCallback(HeuristicCallback):
    """
    Call initialize() before using this callback.

    HeuristicCallback is called intermittently during B&B to generate new feasible
    solutions by rounding, and to improve the quality of existing feasible solutions
    by polishing. Specific heuristics include:

    - Sequential Rounding on the continuous solution from the surrogate LP (only if there has been a change in the
      lower bound). Requires settings['round_flag'] = True. If settings['polish_after_rounding'] = True, then the
      rounded solutions are polished using DCD.

    - DCD Polishing: Polishes integer solutions in polish_queue using DCD. Requires settings['polish_flag'] = True.

    Optional Functionality:

    - Feasible solutions are passed to LazyCutConstraintCallback via cut_queue

    Known issues:

    - Sometimes CPLEX does not return an integer feasible solution (in which case we correct this manually)
    """

    def initialize(self, indices, control, settings, cut_queue, polish_queue, get_objval, get_L0_norm, is_feasible, polishing_handle, rounding_handle):

        #todo: add basic assertions to make sure that nothing weird is going on
        assert isinstance(indices, dict)
        assert isinstance(control, dict)
        assert isinstance(settings, dict)
        assert isinstance(cut_queue, FastSolutionPool)
        assert isinstance(polish_queue, FastSolutionPool)
        assert callable(get_objval)
        assert callable(get_L0_norm)
        assert callable(is_feasible)
        assert callable(polishing_handle)
        assert callable(rounding_handle)

        self.rho_idx = indices['rho']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']
        self.indices = indices
        self.previous_lowerbound = 0.0
        self.stats = control
        self.settings = settings

        self.round_flag = settings['round_flag']
        self.polish_rounded_solutions = settings['polish_rounded_solutions']
        self.polish_flag = settings['polish_flag']
        self.polish_queue = polish_queue

        self.cut_queue = cut_queue  # pointer to cut_queue

        self.rounding_tolerance = float(1.0 + settings['rounding_tolerance'])
        self.rounding_start_cuts = settings['rounding_start_cuts']
        self.rounding_stop_cuts = settings['rounding_stop_cuts']
        self.rounding_stop_gap = settings['rounding_stop_gap']
        self.rounding_start_gap = settings['rounding_start_gap']

        self.polishing_tolerance = float(1.0 + settings['polishing_tolerance'])
        self.polishing_start_cuts = settings['polishing_start_cuts']
        self.polishing_stop_cuts = settings['polishing_stop_cuts']
        self.polishing_stop_gap = settings['polishing_stop_gap']
        self.polishing_start_gap = settings['polishing_start_gap']
        self.polishing_max_solutions = settings['polishing_max_solutions']
        self.polishing_max_runtime = settings['polishing_max_runtime']

        self.get_objval = get_objval
        self.get_L0_norm = get_L0_norm
        self.is_feasible = is_feasible
        self.polishing_handle = polishing_handle
        self.rounding_handle = rounding_handle

        return


    def update_heuristic_flags(self, n_cuts, relative_gap):

        # keep on rounding?
        keep_rounding = (self.rounding_start_cuts <= n_cuts <= self.rounding_stop_cuts) and \
                        (self.rounding_stop_gap <= relative_gap <= self.rounding_start_gap)

        # keep on polishing?
        keep_polishing = (self.polishing_start_cuts <= n_cuts <= self.polishing_stop_cuts) and \
                         (self.polishing_stop_gap <=  relative_gap <= self.polishing_start_gap)

        self.round_flag &= keep_rounding
        self.polish_flag &= keep_polishing
        self.polish_rounded_solutions &= self.round_flag

        return


    def __call__(self):
        # todo write rounding/polishing as separate function calls

        # print_log('in heuristic callback')
        if not (self.round_flag or self.polish_flag):
            return

        callback_start_time = time.time()
        self.stats['heuristic_callback_times_called'] += 1
        self.stats['upperbound'] = self.get_incumbent_objective_value()
        self.stats['lowerbound'] = self.get_best_objective_value()
        self.stats['relative_gap'] = self.get_MIP_relative_gap()

        # check if lower bound was updated since last call
        lowerbound_update = self.previous_lowerbound < self.stats['lowerbound']
        if lowerbound_update:
            self.previous_lowerbound = self.stats['lowerbound']

        # check if incumbent solution has been updated
        # if incumbent solution is not integer, then recast as integer and update objective value manually
        if self.has_incumbent():
            cplex_incumbent = np.array(self.get_incumbent_values(self.rho_idx))
            cplex_rounding_issue = not is_integer(cplex_incumbent)
            if cplex_rounding_issue:
                cplex_incumbent = cast_to_integer(cplex_incumbent)

            incumbent_update = not np.array_equal(cplex_incumbent, self.stats['incumbent'])

            if incumbent_update:
                self.stats['incumbent'] = cplex_incumbent
                self.stats['n_incumbent_updates'] += 1
                if cplex_rounding_issue:
                    self.stats['upperbound'] = self.get_objval(cplex_incumbent)

        # update flags on whether or not to keep rounding / polishing
        self.update_heuristic_flags(n_cuts = self.stats['n_cuts'], relative_gap = self.stats['relative_gap'])

        #variables to store best objective value / solution from heuristics
        best_objval = float('inf')
        best_solution = None

        # run sequential rounding if lower bound was updated since the last call
        if self.round_flag and lowerbound_update:

            rho_cts = np.array(self.get_values(self.rho_idx))
            zero_idx_rho_ceil = np.equal(np.ceil(rho_cts), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho_cts), 0)
            cannot_round_to_zero = np.logical_not(np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor))
            min_l0_norm = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            max_l0_norm = np.count_nonzero(rho_cts[self.L0_reg_ind])
            rounded_solution_is_feasible = (min_l0_norm < self.stats['bounds'].L0_max and max_l0_norm > self.stats['bounds'].L0_min)

            if rounded_solution_is_feasible:

                rounding_cutoff = self.rounding_tolerance * self.stats['upperbound']
                rounding_start_time = time.time()
                rounded_solution, rounded_objval, early_stop = self.rounding_handle(rho_cts, rounding_cutoff)
                self.stats['total_round_time'] += time.time() - rounding_start_time
                self.stats['n_rounded'] += 1

                # round solution if sequential rounding did not stop early
                if not early_stop:

                    if self.settings['add_cuts_at_heuristic_solutions']:
                        self.cut_queue.add(rounded_objval, rounded_solution)

                    if self.is_feasible(rounded_solution):
                        best_solution = rounded_solution
                        best_objval = rounded_objval

                    if self.polish_rounded_solutions:
                        current_upperbound = min(rounded_objval, self.stats['upperbound'])
                        polishing_cutoff = current_upperbound * self.polishing_tolerance

                        if rounded_objval < polishing_cutoff:
                            start_time = time.time()
                            polished_solution, _, polished_objval = self.polishing_handle(rounded_solution)
                            self.stats['total_round_then_polish_time'] += time.time() - start_time
                            self.stats['n_rounded_then_polished'] += 1

                            if self.settings['add_cuts_at_heuristic_solutions']:
                                self.cut_queue.add(polished_objval, polished_solution)

                            if self.is_feasible(polished_solution):
                                best_solution = polished_solution
                                best_objval = polished_objval

        # polish solutions in polish_queue or that were produced by rounding
        if self.polish_flag and len(self.polish_queue) > 0:

            #get current upperbound
            current_upperbound = min(best_objval, self.stats['upperbound'])
            polishing_cutoff = self.polishing_tolerance * current_upperbound
            self.polish_queue.filter_sort_unique(max_objval = polishing_cutoff)

            if len(self.polish_queue) > 0:
                polished_queue = FastSolutionPool(self.polish_queue.P)
                polish_time = 0
                n_polished = 0
                for objval, solution in zip(self.polish_queue.objvals, self.polish_queue.solutions):

                    if objval >= polishing_cutoff:
                        break

                    polish_start_time = time.time()
                    polished_solution, _, polished_objval = self.polishing_handle(solution)
                    polish_time += time.time() - polish_start_time
                    n_polished += 1

                    # update cutoff whenever the solution returns an infeasible solutions
                    if self.is_feasible(polished_solution):
                        polished_queue.add(polished_objval, polished_solution)
                        current_upperbound = min(polished_objval, polished_objval)
                        polishing_cutoff = self.polishing_tolerance * current_upperbound

                    if polish_time > self.polishing_max_runtime:
                        break

                    if n_polished > self.polishing_max_solutions:
                        break

                self.polish_queue.clear()
                self.stats['total_polish_time'] += polish_time
                self.stats['n_polished'] += n_polished

                if self.settings['add_cuts_at_heuristic_solutions']:
                    self.cut_queue.add(polished_queue.objvals, polished_queue.solutions)

                # check if the best polished solution will improve the queue
                polished_queue.filter_sort_unique(max_objval = best_objval)
                if len(polished_queue) > 0:
                    best_objval, best_solution = polished_queue.get_best_objval_and_solution()

        # if heuristics produces a better solution then update the incumbent
        heuristic_update = best_objval < self.stats['upperbound']
        if heuristic_update:
            self.stats['n_heuristic_updates'] += 1
            proposed_solution, proposed_objval = convert_to_risk_slim_cplex_solution(indices = self.indices, rho = best_solution, objval = best_objval)
            self.set_solution(solution = proposed_solution, objective_value = proposed_objval)

        self.stats['total_heuristic_callback_time'] += time.time() - callback_start_time
        #print_log('left heuristic callback')
        return




