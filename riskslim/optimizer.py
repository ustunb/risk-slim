"""RiskSLIM optimizer."""

import time
import numpy as np
import warnings
from copy import copy

from cplex.exceptions import CplexError

from riskslim.utils import check_data, validate_settings, print_log
from riskslim.defaults import DEFAULT_LCPA_SETTINGS
from riskslim.coefficient_set import CoefficientSet, get_score_bounds
from riskslim.mip import add_mip_starts, create_risk_slim, set_cplex_mip_parameters
from riskslim.solution_pool import SolutionPool, FastSolutionPool
from riskslim.data import Bounds, Stats
from riskslim.heuristics import discrete_descent, sequential_rounding
from riskslim.bound_tightening import chained_updates
from riskslim.warmstart import (
    run_standard_cpa,
    round_solution_pool,
    sequential_round_solution_pool,
    discrete_descent_solution_pool,
)
from riskslim.callbacks import LossCallback, PolishAndRoundCallback


class RiskSLIMOptimizer:
    """RiskSLIM optimizer object.

    Parameters
    ----------
    L0_min : int, optional, default: None
        Minimum number of regularized coefficients.
        None defaults to zero.
    L0_max : int, optional, default: None
        Maximum number of regularized coefficients.
        None defaults to length of input variables.
    rho_min : float or 1d array, optional, default: -5.
        Minimum coefficient.
    rho_max : float or 1d array, optional, default: 5.
        Maximum coefficient.
    c0_value : 1d array or float, optional, default: 1e-6
        L0-penalty for all parameters when an integer or for each parameter
        separately when an array.
    max_abs_offset : float, optional, default: None
        Maximum absolute value of intercept. This may be specificed as the first value
        of min_coef and max_coef. However, if min_coef and max_coef are floats, this parameter
        provides a convenient way to set bounds on the offset.
    vtype : str or list of str, optional, default: "I"
        Variable types for coefficients. Must be either "I" for integers or "C" for floats.
    settings, dict, optional, defaults: None
        Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
        None defaults to settings defined in riskslim.defaults.
    coefficient_set : riskslim.coefficient_set.CoefficientSet, optional, default: None
        Contraints (bounds) on coefficients of input variables.
        If None, this is constructed based on values passed to other initalization kwargs.
        If not None, other kwargs may be overwritten.
    verbose : bool, optional, default: True
        Prints out log information if True, supresses if False.
    """
    #DEFAULT_SETTINGS = {
        # ''
        # }

    def __init__(
        self, L0_min=None, L0_max=None, rho_min=-5., rho_max=5., c0_value=1e-6,
        max_abs_offset=None, vtype="I", settings=None, coef_set=None, verbose=True
    ):

        # Empty fields
        self.fitted = False
        self.has_warmstart = False
        self._default_print_flag = False
        self.initial_cuts = None

        # Settings
        settings = {} if settings is None else settings

        if 'display_cplex_progress' not in settings.keys():
            settings['display_cplex_progress'] = verbose

        (
            self.settings,
            self.cplex_settings,
            self.warmstart_settings,
        ) = self._parse_settings(settings, defaults=DEFAULT_LCPA_SETTINGS)

        self.loss_computation = self.settings["loss_computation"]
        self.verbose = verbose
        self.print_log = lambda msg: print_log(msg, print_flag=self.verbose)

        # Coefficient constraints
        self.L0_min = L0_min
        self.L0_max = L0_max
        self.coef_set = coef_set
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho = None
        self.c0_value = c0_value
        self.vtype = vtype
        self.max_abs_offset = max_abs_offset

        if self.coef_set is not None:
            self._init_coeffs()
        else:
            self.L0_reg_ind = None
            self.C_0 = None
            self.C_0_nnz = None
            self.variable_names = None

        # Features and labels
        self.X = None
        self.y = None


    def optimize(self, X, y, variable_names=None, outcome_name=None, sample_weights=None):
        """Optimize RiskSLIM.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        variable_names : list of str, optional, default: None
            Names of each features. Only needed if coefficients is not passed on
            initalization. None defaults to generic variable names.
        outcome_name : str, optional, default: None
            Name of the output class.
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        """
        # Set data attributes
        self.X = X
        self.y = y
        self.variable_names = variable_names
        self.outcome_name = outcome_name
        self.sample_weights = sample_weights

        # Initalize fitting procedure
        self._init_fit()

        if self.max_abs_offset is not None:
            # Set offset bounds
            self.coef_set.update_intercept_bounds(
                X=self.X, y=self.y, max_offset=self.max_abs_offset
            )

        # Initalize MIP
        self._init_mip()

        # Run warmstart procedure if it has not been run yet
        if self.settings["initialization_flag"] and not self.has_warmstart:
            self.warmstart()

        # Set cplex parameters and runtime
        cpx = self.mip
        cpx = set_cplex_mip_parameters(
            cpx,
            self.cplex_settings,
            display_cplex_progress=self.settings["display_cplex_progress"],
        )
        cpx.parameters.timelimit.set(self.settings["max_runtime"])

        # Initialize solution queues
        self.cut_queue = FastSolutionPool(self.n_variables)
        self.polish_queue = FastSolutionPool(self.n_variables)

        if not self.fitted:
            self._init_callbacks()

        # Initialize solution pool
        if len(self.pool) > 0:
            # Initialize using the polish_queue when possible to avoid bugs with
            #   the CPLEX MIPStart interface
            if self.settings["polish_flag"]:
                self.polish_queue.add(self.pool.objvals[0], self.pool.solutions[0])
            else:
                cpx = add_mip_starts(
                    cpx,
                    self.mip_indices,
                    self.pool,
                    mip_start_effort_level=cpx.MIP_starts.effort_level.repair,
                )

            if self.settings["add_cuts_at_heuristic_solutions"] and len(self.pool) > 1:
                self.cut_queue.add(self.pool.objvals[1:], self.pool.solutions[1:])

        # Solve riskslim optimization problem
        start_time = time.time()
        cpx.solve()
        self.stats.total_run_time = time.time() - start_time
        self.fitted = True
        self.rho = self.solution_info['solution']


    def warmstart(self, warmstart_settings=None):
        """Run an initialization routine to speed up RiskSLIM.

        Parameters
        ----------
        warmstart_settings : dict
            Warmstart settings to overwrite settings passed on initalization.
        """
        if warmstart_settings is None:
            settings = self.warmstart_settings
        else:
            settings = validate_settings(
                warmstart_settings, defaults=self.warmstart_settings
            )

        settings["type"] = "cvx"

        # Construct LP relaxation
        lp_settings = dict(self.mip_settings)
        lp_settings["relax_integer_variables"] = True
        cpx, cpx_indices = create_risk_slim(coef_set=self.coef_set, settings=lp_settings)
        cpx = set_cplex_mip_parameters(
            cpx,
            self.cplex_settings,
            display_cplex_progress=settings["display_cplex_progress"],
        )

        # Solve RiskSLIM LP using standard CPA
        stats, cuts, pool = run_standard_cpa(
            cpx=cpx,
            cpx_indices=cpx_indices,
            compute_loss=self.compute_loss_real,
            compute_loss_cut=self.compute_loss_cut_real,
            settings=settings,
            print_flag=self.verbose,
        )

        # Update cuts
        self.print_log("warmstart CPA produced %d cuts" % len(cuts["coefs"]))
        self.initial_cuts = cuts

        # Update bounds
        bounds = chained_updates(
            self.bounds, self.C_0_nnz, new_objval_at_relaxation=stats["lowerbound"]
        )

        constraints = {
            "min_size": self.L0_min,
            "max_size": self.L0_max,
            "coef_set": self.coef_set,
        }

        def rounded_model_size_is_ok(rho):
            zero_idx_rho_ceil = np.equal(np.ceil(rho), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho), 0)
            cannot_round_to_zero = np.logical_not(
                np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor)
            )
            rounded_rho_L0_min = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            rounded_rho_L0_max = np.count_nonzero(rho[self.L0_reg_ind])
            return (
                rounded_rho_L0_min >= self.L0_min >= 0
                and rounded_rho_L0_max <= self.L0_max
            )

        pool = pool.remove_infeasible(rounded_model_size_is_ok).distinct().sort()

        if len(pool) == 0:
            self.print_log("all CPA solutions are infeasible")

        # Round CPA solutions
        if settings["use_rounding"] and len(pool) > 0:
            self.print_log("running naive rounding on %d solutions" % len(pool))
            self.print_log("best objective value: %1.4f" % np.min(pool.objvals))
            rnd_pool, _, _ = round_solution_pool(
                pool,
                constraints,
                max_runtime=settings["rounding_max_runtime"],
                max_solutions=settings["rounding_max_solutions"],
            )
            rnd_pool = rnd_pool.compute_objvals(self.get_objval).remove_infeasible(
                self.is_feasible
            )
            self.print_log("rounding produced %d integer solutions" % len(rnd_pool))

            if len(rnd_pool) > 0:
                pool.append(rnd_pool)
                self.print_log(
                    "best objective value is %1.4f" % np.min(rnd_pool.objvals)
                )

        # Sequentially round CPA solutions
        if settings["use_sequential_rounding"] and len(pool) > 0:
            self.print_log("running sequential rounding on %d solutions" % len(pool))
            self.print_log("best objective value: %1.4f" % np.min(pool.objvals))

            sqrnd_pool, _, _ = sequential_round_solution_pool(
                pool=pool,
                Z=self.Z,
                C_0=self.C_0,
                compute_loss_from_scores_real=self.compute_loss_from_scores_real,
                get_L0_penalty=self.get_L0_penalty,
                max_runtime=settings["sequential_rounding_max_runtime"],
                max_solutions=settings["sequential_rounding_max_solutions"],
                objval_cutoff=self.bounds.objval_max,
            )

            sqrnd_pool = sqrnd_pool.remove_infeasible(self.is_feasible)
            self.print_log(
                "sequential rounding produced %d integer solutions" % len(sqrnd_pool)
            )

            if len(sqrnd_pool) > 0:
                pool = pool.append(sqrnd_pool)
                self.print_log("best objective value: %1.4f" % np.min(pool.objvals))

        # Polish rounded solutions
        if settings["polishing_after"] and len(pool) > 0:
            self.print_log("polishing %d solutions" % len(pool))
            self.print_log("best objective value: %1.4f" % np.min(pool.objvals))
            dcd_pool, _, _ = discrete_descent_solution_pool(
                pool=pool,
                Z=self.Z,
                C_0=self.C_0,
                constraints=constraints,
                compute_loss_from_scores=self.compute_loss_from_scores,
                get_L0_penalty=self.get_L0_penalty,
                max_runtime=settings["polishing_max_runtime"],
                max_solutions=settings["polishing_max_solutions"],
            )

            dcd_pool = dcd_pool.remove_infeasible(self.is_feasible)
            if len(dcd_pool) > 0:
                self.print_log(
                    "polishing produced %d integer solutions" % len(dcd_pool)
                )
                pool.append(dcd_pool)

        # Remove solutions that are not feasible, not integer
        if len(pool) > 0:
            pool = pool.remove_nonintegral().distinct().sort()

        # Update upper and lower bounds
        self.print_log("initialization produced %1.0f feasible solutions" % len(pool))

        if len(pool) > 0:
            bounds = chained_updates(
                bounds, self.C_0_nnz, new_objval_at_feasible=np.min(pool.objvals)
            )
            self.print_log("best objective value: %1.4f" % np.min(pool.objvals))

        self.pool.append(pool)

        # Update bounds
        self.bounds = copy(bounds)
        self.has_warmstart = True

    #### properties ####

    @property
    def solution(self):
        """Returns CPLEX solution.

        Returns
        -------
        cplex SolutionInterface
        """
        # todo add wrapper if solution does not exist
        return self.mip.solution

    @property
    def solution_info(self):
        """Returns information associated with the current best solution for the mip.

        Returns
        -------
        info : dict
            Contains best solution info.
        """
        # Record mip solution statistics
        solution = self.solution

        try:
            self.stats.incumbent = np.array(
                solution.get_values(self.mip_indices["rho"])
            )
            self.stats.upperbound = solution.get_objective_value()
            self.stats.lowerbound = solution.MIP.get_best_objective()
            self.stats.relative_gap = solution.MIP.get_mip_relative_gap()
            self.stats.found_solution = True
        except CplexError:
            self.stats.found_solution = False

        self.stats.cplex_status = solution.get_status_string()
        self.stats.total_callback_time = (
            self.stats.total_cut_callback_time
            + self.stats.total_heuristic_callback_time
        )
        self.stats.total_solver_time = (
            self.stats.total_run_time - self.stats.total_callback_time
        )
        self.stats.total_data_time = (
            self.stats.total_cut_time
            + self.stats.total_polish_time
            + self.stats.total_round_time
            + self.stats.total_round_then_polish_time
        )

        # Output for Model
        info = {
            "c0_value": self.c0_value,
            #
            "solution": self.stats.incumbent,
            "objective_value": self.get_objval(self.stats.incumbent)
            if self.stats.found_solution
            else float("inf"),
            "loss_value": self.compute_loss(self.stats.incumbent)
            if self.stats.found_solution
            else float("inf"),
            "optimality_gap": self.stats.relative_gap
            if self.stats.found_solution
            else float("inf"),
            #
            "run_time": self.stats.total_run_time,
            "solver_time": self.stats.total_solver_time,
            "callback_time": self.stats.total_callback_time,
            "data_time": self.stats.total_data_time,
            "nodes_processed": self.stats.nodes_processed,
        }

        return info

    @property
    def coefficients(self):
        """
        Returns
        -------
        coefs : np.ndarray
            C
            oefficients of the linear classifier
        """
        s = self.solution
        if s.is_primal_feasible():
            coefs = np.array(s.get_values(self.mip_indices["rho"]))
        else:
            coefs = np.repeat(np.nan, self.n_variables)
        return coefs


    # helper functions
    def is_feasible(self, rho):
        """Ensure constraints are obeyed.

        Parameters
        ---------
        """
        return (
            np.all(self.rho_max >= rho)
            and np.all(self.rho_min <= rho)
            and (self.L0_min <= np.count_nonzero(rho[self.L0_reg_ind]) <= self.L0_max)
        )

    ### initialization ####
    def _parse_settings(self, settings, defaults):
        """Parse settings and separete into warmstart, lcpa, cplex."""

        settings = validate_settings(settings, defaults)
        warmstart_settings = {
            k[5:]: settings[k] for k in settings if k.startswith("init_")
        }
        cplex_settings = {
            k[6:]: settings[k] for k in settings if k.startswith("cplex_")
        }
        lcpa_settings = {
            k: settings[k]
            for k in settings
            if settings
            if not k.startswith(("init_", "cplex_"))
        }
        return lcpa_settings, cplex_settings, warmstart_settings


    def _init_loss_computation(self, w_pos=1.0):
        """Initalize loss functions."""
        # todo check if fast/lookup loss is installed
        assert self.loss_computation in ["normal", "weighted", "fast", "lookup"]

        use_weighted = False
        use_lookup_table = (
            isinstance(self.coef_set, CoefficientSet) and self._integer_data
        )

        if self.sample_weights is not None:
            sample_weights = self._init_training_weights(w_pos=w_pos)
            use_weighted = not np.all(np.equal(sample_weights, 1.0))

        if use_weighted:
            final_loss_computation = "weighted"
        elif use_lookup_table and self.loss_computation == "lookup":
            final_loss_computation = "lookup"
        elif self.loss_computation == "normal":
            final_loss_computation = "normal"
        else:
            final_loss_computation = "fast"

        if final_loss_computation != self.loss_computation:
            warnings.warn(
                "switching loss computation from %s to %s"
                % (self.loss_computation, final_loss_computation)
            )

        self.loss_computation = final_loss_computation

        if final_loss_computation == "normal":
            import riskslim.loss_functions.log_loss as lf

            self.Z = np.require(self.Z, requirements=["C"])
            self.compute_loss = lambda rho: lf.log_loss_value(self.Z, rho)
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self.Z, rho)
            self.compute_loss_from_scores = (
                lambda scores: lf.log_loss_value_from_scores(scores)
            )

        elif final_loss_computation == "fast":
            import riskslim.loss_functions.fast_log_loss as lf

            self.Z = np.require(self.Z, requirements=["F"])
            self.compute_loss = lambda rho: lf.log_loss_value(
                self.Z, np.asfortranarray(rho)
            )
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(
                self.Z, np.asfortranarray(rho)
            )
            self.compute_loss_from_scores = (
                lambda scores: lf.log_loss_value_from_scores(scores)
            )

        elif final_loss_computation == "weighted":
            import riskslim.loss_functions.log_loss_weighted as lf

            self.Z = np.require(self.Z, requirements=["C"])
            total_sample_weights = np.sum(sample_weights)
            self.compute_loss = lambda rho: lf.log_loss_value(
                self.Z, sample_weights, total_sample_weights, rho
            )
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(
                self.Z, sample_weights, total_sample_weights, rho
            )
            self.compute_loss_from_scores = (
                lambda scores: lf.log_loss_value_from_scores(
                    sample_weights, total_sample_weights, scores
                )
            )

        elif final_loss_computation == "lookup":
            import riskslim.loss_functions.lookup_log_loss as lf

            self.Z = np.require(self.Z, requirements=["F"], dtype=np.float64)

            s_min, s_max = get_score_bounds(
                Z_min=np.min(self.Z, axis=0),
                Z_max=np.max(self.Z, axis=0),
                rho_lb=self.coef_set.lb,
                rho_ub=self.coef_set.ub,
                L0_reg_ind=np.array(self.coef_set.c0) == 0.0,
                L0_max=self.L0_max,
            )

            self.print_log("%d rows in lookup table" % (s_max - s_min + 1))
            (
                loss_value_tbl,
                prob_value_tbl,
                tbl_offset,
            ) = lf.get_loss_value_and_prob_tables(s_min, s_max)
            self.compute_loss = lambda rho: lf.log_loss_value(
                self.Z, rho, loss_value_tbl, tbl_offset
            )
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(
                self.Z, rho, loss_value_tbl, prob_value_tbl, tbl_offset
            )
            self.compute_loss_from_scores = (
                lambda scores: lf.log_loss_value_from_scores(
                    scores, loss_value_tbl, tbl_offset
                )
            )

        # real loss functions
        if final_loss_computation == "lookup":
            import riskslim.loss_functions.fast_log_loss as lfr

            self.compute_loss_real = lambda rho: lfr.log_loss_value(self.Z, rho)
            self.compute_loss_cut_real = lambda rho: lfr.log_loss_value_and_slope(
                self.Z, rho
            )
            self.compute_loss_from_scores_real = (
                lambda scores: lfr.log_loss_value_from_scores(scores)
            )

        else:
            self.compute_loss_real = self.compute_loss
            self.compute_loss_cut_real = self.compute_loss_cut
            self.compute_loss_from_scores_real = self.compute_loss_from_scores

    def _init_training_weights(self, w_pos=1.0, w_neg=1.0, w_total_target=2.0):
        """Initialize weights.

        Parameters
        ----------
        w_pos : float
            Positive scalar showing relative weight on examples where Y = +1
        w_neg : float
            Positive scalar showing relative weight on examples where Y = -1

        Returns
        -------
        training_weights : 1d-array
            A vector of N normalized training weights for all points in the training data.
        """

        # Process class weights
        assert np.isfinite(w_pos), "w_pos must be finite"
        assert np.isfinite(w_neg), "w_neg must be finite"
        assert w_pos > 0.0, "w_pos must be strictly positive"
        assert w_neg > 0.0, "w_neg must be strictly positive"
        w_total = w_pos + w_neg
        w_pos = w_total_target * (w_pos / w_total)
        w_neg = w_total_target * (w_neg / w_total)

        # Process case weights
        y = self.y.flatten()
        N = len(y)
        pos_ind = y == 1

        if self.sample_weights is None:
            training_weights = np.ones(N)
        else:
            training_weights = self.sample_weights.flatten()

        # Normalization
        training_weights = N * (training_weights / sum(training_weights))
        training_weights[pos_ind] *= w_pos
        training_weights[~pos_ind] *= w_neg

        return training_weights


    def _init_coeffs(self):
        """Initialize coefficient constraints.

        Notes
        -----
        If coef_set is passed during initalization, this is ran at initalization time.
        Otherwise, this will be ran when calling the fit method.
        """
        # Coefficients
        self.rho_min = np.array(self.coef_set.lb)
        self.rho_max = np.array(self.coef_set.ub)

        # Regularization parameter
        c0_value = self.c0_value
        assert c0_value > 0.0, "C0 should be positive"
        self.c0_value = c0_value

        # Vectorized regularization parameters
        self.L0_reg_ind = np.isnan(self.coef_set.c0) + self.coef_set.c0 != 0.0
        self.C_0 = np.array(self.coef_set.c0)
        self.C_0[self.L0_reg_ind] = c0_value
        self.C_0_nnz = self.C_0[self.L0_reg_ind]

        # Model size constraints
        L0_max = np.inf if self.L0_max is None else self.L0_max
        L0_min = 0 if self.L0_min is None else self.L0_min
        self.L0_max = np.minimum(L0_max, np.sum(self.L0_reg_ind))
        self.L0_min = np.maximum(L0_min, 0)

        # Variable names
        self.variable_names = self.coef_set.variable_names


    def _init_fit(self):
        """Pre-fit initialization routine."""
        # Initialize data dict
        if self.variable_names is None:
            self.variable_names = [
                f"Variable_{str(i).zfill(3)}" for i in range(len(self.X[0]) - 1)
            ]
            self.variable_names.insert(0, "(Intercept)")

        # Initialize coefficients if not given on initialization
        if self.coef_set is None:
            self.coef_set = CoefficientSet(
                self.variable_names,
                lb=self.rho_min,
                ub=self.rho_max,
                c0=self.c0_value,
                vtype=self.vtype,
                print_flag=self.verbose
            )
            self._init_coeffs()

        # Check data types and shapes
        check_data(
            self.X, self.y, self.variable_names, self.outcome_name, self.sample_weights
        )

        self.Z = (self.X * self.y).astype(np.float64)
        self._integer_data = np.all(self.Z == np.require(self.Z, dtype=np.int_))
        self.n_variables = self.Z.shape[1]

        # Function handles
        self._init_loss_computation()
        self.get_L0_norm = lambda rho: np.count_nonzero(rho[self.L0_reg_ind])
        self.get_L0_penalty = lambda rho: np.sum(
            self.C_0_nnz * (rho[self.L0_reg_ind] != 0.0)
        )
        self.get_alpha = lambda rho: np.array(
            abs(rho[self.L0_reg_ind]) > 0.0, dtype=np.float_
        )
        self.get_L0_penalty_from_alpha = lambda alpha: np.sum(self.C_0_nnz * alpha)
        self.get_objval = lambda rho: self.compute_loss(rho) + np.sum(
            self.C_0_nnz * (rho[self.L0_reg_ind] != 0.0)
        )

        # Bounds
        self.bounds = Bounds(L0_min=self.L0_min, L0_max=self.L0_max)
        self.bounds.loss_min, self.bounds.loss_max = self._init_loss_bounds()

        # Solution
        self.pool = SolutionPool(self.n_variables)

        # Check if trivial solution is feasible, if so add it to the pool and update bounds
        trivial_solution = np.zeros(self.n_variables, dtype=np.float64)
        if self.is_feasible(trivial_solution):
            trivial_objval = self.compute_loss(trivial_solution)
            if self.settings["initial_bound_updates"]:
                self.bounds.objval_max = min(self.bounds.objval_max, trivial_objval)
                self.bounds.loss_max = min(self.bounds.loss_max, trivial_objval)
                self.bounds = chained_updates(self.bounds, self.C_0_nnz)
            self.pool.add(objvals=trivial_objval, solutions=trivial_solution)


    def _init_mip(self):
        """Initalize a CPLEX MIP solver."""
        # Setup mip
        mip_settings = {
            "C_0": self.c0_value,
            "coef_set": self.coef_set,
            "tight_formulation": self.settings["tight_formulation"],
            "drop_variables": self.settings["drop_variables"],
            "include_auxillary_variable_for_L0_norm": self.settings[
                "include_auxillary_variable_for_L0_norm"
            ],
            "include_auxillary_variable_for_objval": self.settings[
                "include_auxillary_variable_for_objval"
            ],
        }

        mip_settings.update(self.bounds.asdict())
        self.mip_settings = mip_settings

        # Setup risk slim mip
        self.mip, self.mip_indices = create_risk_slim(
            coef_set=self.coef_set, settings=self.mip_settings
        )
        self.mip_indices.update(
            {
                "C_0_nnz": self.C_0_nnz,
                "L0_reg_ind": self.L0_reg_ind,
            }
        )

        self.stats = Stats(
            incumbent=np.full(self.X.shape[1], np.nan), bounds=self.bounds
        )


    def _init_callbacks(self):
        """Initializes callback functions"""

        loss_cb = self.mip.register_callback(LossCallback)

        loss_cb.initialize(
            indices=self.mip_indices,
            stats=self.stats,
            settings=self.settings,
            compute_loss_cut=self.compute_loss_cut,
            get_alpha=self.get_alpha,
            get_L0_penalty_from_alpha=self.get_L0_penalty_from_alpha,
            initial_cuts=self.initial_cuts,
            cut_queue=self.cut_queue,
            polish_queue=self.polish_queue,
            verbose=self.verbose,
        )

        self.loss_callback = loss_cb

        # add heuristic callback if rounding or polishing
        if self.settings["round_flag"] or self.settings["polish_flag"]:
            heuristic_cb = self.mip.register_callback(PolishAndRoundCallback)
            active_set_flag = self.L0_max <= self.n_variables
            polisher = lambda rho: discrete_descent(
                rho,
                self.Z,
                self.C_0,
                self.rho_max,
                self.rho_min,
                self.get_L0_penalty,
                self.compute_loss_from_scores,
                active_set_flag,
            )

            rounder = lambda rho, cutoff: sequential_rounding(
                rho,
                self.Z,
                self.C_0,
                self.compute_loss_from_scores_real,
                self.get_L0_penalty,
                cutoff,
            )

            heuristic_cb.initialize(
                indices=self.mip_indices,
                control=self.stats,
                settings=self.settings,
                cut_queue=self.cut_queue,
                polish_queue=self.polish_queue,
                get_objval=self.get_objval,
                get_L0_norm=self.get_L0_norm,
                is_feasible=self.is_feasible,
                polishing_handle=polisher,
                rounding_handle=rounder,
            )

            self.heuristic_callback = heuristic_cb

    def _init_loss_bounds(self):
        # min value of loss = log(1+exp(-score)) occurs at max score for each point
        # max value of loss = loss(1+exp(-score)) occurs at min score for each point

        # get maximum number of regularized coefficients
        num_max_reg_coefs = self.L0_max

        # calculate the smallest and largest score that can be attained by each point
        scores_at_lb = self.Z * self.rho_min
        scores_at_ub = self.Z * self.rho_max
        max_scores_matrix = np.maximum(scores_at_ub, scores_at_lb)
        min_scores_matrix = np.minimum(scores_at_ub, scores_at_lb)
        assert np.all(max_scores_matrix >= min_scores_matrix)

        # for each example, compute max sum of scores from top reg coefficients
        max_scores_reg = max_scores_matrix[:, self.L0_reg_ind]
        max_scores_reg = -np.sort(-max_scores_reg, axis=1)
        max_scores_reg = max_scores_reg[:, 0:num_max_reg_coefs]
        max_score_reg = np.sum(max_scores_reg, axis=1)

        # for each example, compute max sum of scores from no reg coefficients
        max_scores_no_reg = max_scores_matrix[:, ~self.L0_reg_ind]
        max_score_no_reg = np.sum(max_scores_no_reg, axis=1)

        # max score for each example
        max_score = max_score_reg + max_score_no_reg

        # for each example, compute min sum of scores from top reg coefficients
        min_scores_reg = min_scores_matrix[:, self.L0_reg_ind]
        min_scores_reg = np.sort(min_scores_reg, axis=1)
        min_scores_reg = min_scores_reg[:, 0:num_max_reg_coefs]
        min_score_reg = np.sum(min_scores_reg, axis=1)

        # for each example, compute min sum of scores from no reg coefficients
        min_scores_no_reg = min_scores_matrix[:, ~self.L0_reg_ind]
        min_score_no_reg = np.sum(min_scores_no_reg, axis=1)

        min_score = min_score_reg + min_score_no_reg
        assert np.all(max_score >= min_score)

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
