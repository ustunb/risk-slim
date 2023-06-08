"""RiskSLIM optimizer."""

import time
import numpy as np
from warnings import warn
from copy import copy

import cplex
from cplex.exceptions import CplexError

from riskslim.utils import Stats, check_data, validate_settings, print_log
from riskslim.defaults import DEFAULT_LCPA_SETTINGS
from riskslim.coefficient_set import CoefficientSet, get_score_bounds
from riskslim.mip import add_mip_starts, create_risk_slim, set_cplex_mip_parameters
from riskslim.solution_pool import SolutionPool, FastSolutionPool
from riskslim.heuristics import discrete_descent, sequential_rounding
from riskslim.bound_tightening import Bounds, chained_updates
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
    min_size : int, optional, default: None
        Minimum number of regularized coefficients.
        None defaults to zero.
    max_size : int, optional, default: None
        Maximum number of regularized coefficients.
        None defaults to length of input variables.
    min_coef : float or 1d array, optional, default: -5.
        Minimum coefficient.
    max_coef : float or 1d array, optional, default: 5.
        Maximum coefficient.
    c0_value : 1d array or float, optional, default: 1e-6
        L0-penalty for all parameters when an integer or for each parameter
        separately when an array.
    max_abs_offset : float, optional, default: None
        Maximum absolute value of intercept. This may be specificed as the first value
        of min_coef and max_coef. However, if min_coef and max_coef are floats, this parameter
        provides a convenient way to set bounds on the offset.
    variable_names : list of str, optional, default: None
            Names of each features. Only needed if coefficients is not passed on
            initalization. None defaults to generic variable names.
    outcome_name : str, optional, default: None
        Name of the output class.
    verbose : bool, optional, default: True
        Prints out log information if True, supresses if False.
    **kwargs
        May include key value pairs:

            "coef_set" : riskslim.coefficient_set.CoefficientSet
                Contraints (bounds) on coefficients of input variables.
                If None, this is constructed based on values passed to other initalization kwargs.
                If not None, other kwargs may be overwritten.

            "vtype" : str or list of str
                Variable types for coefficients.
                Must be either "I" for integers or "C" for floats.

            **settings : unpacked dict
                Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
                Defaults are defined in defaults.DEFAULT_LCPA_SETTINGS.
    """

    def __init__(
            self,
            min_size=None,
            max_size=None,
            min_coef=-5.0,
            max_coef=5.0,
            c0_value=1e-6,
            max_abs_offset=None,
            variable_names=None,
            outcome_name=None,
            verbose=True,
            constraints=None,
            **kwargs
            ):
        # Empty fields
        self.fitted = False
        self.has_warmstart = False
        self._default_print_flag = False
        self.initial_cuts = None

        # Handle kwargs
        self.vtype = kwargs.pop("vtype", None)
        self.coef_set = kwargs.pop("coef_set", None)
        settings = kwargs

        # Parse settings
        if "display_cplex_progress" not in settings.keys():
            settings["display_cplex_progress"] = verbose

        self.settings, self.cplex_settings, self.warmstart_settings = self._parse_settings(settings, defaults = DEFAULT_LCPA_SETTINGS)
        self.loss_computation = self.settings["loss_computation"]
        self.verbose = verbose
        self.log = lambda msg: print_log(msg, print_flag=self.verbose)

        # Coefficient constraints
        self.min_size = min_size
        self.max_size = max_size
        self.min_coef = min_coef
        self.max_coef = max_coef
        self.rho = None
        self.c0_value = c0_value
        self.max_abs_offset = max_abs_offset
        self.constraints = constraints if constraints is not None else []

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

        self.variable_names = variable_names
        self.outcome_name = outcome_name


    def optimize(self, X, y, sample_weights=None):
        """Optimize RiskSLIM.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        """
        # Set data attributes
        self.X = X
        self.y = y
        self.sample_weights = np.ones(X.shape[0]) if sample_weights is None else sample_weights

        #todo: check data attributes

        # Initalize fitting procedure
        self._init_fit()

        if self.max_abs_offset is not None:
            # Set offset bounds
            self.coef_set.update_intercept_bounds(
                    X=self.X, y=self.y, max_offset=self.max_abs_offset
                    )

        # Initalize MIP
        self._init_mip()

        # Add MIP constraints
        for name, var_inds, values, sense, rhs in self.constraints:

            if name is None:
                name = "con_" + str(self.mip.linear_constraints.get_num())

            self.mip.linear_constraints.add(
                    names=[name],
                    lin_expr=[
                        cplex.SparsePair(
                                ind=var_inds,
                                val=values
                                )
                        ],
                    senses=[sense],
                    rhs=[rhs]
                    )

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
        self.rho = self.solution_info["solution"]

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
        cpx, cpx_indices = create_risk_slim(
                coef_set=self.coef_set, settings=lp_settings
                )
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
        self.log("warmstart CPA produced {} cuts".format(len(cuts["coefs"])))
        self.initial_cuts = cuts

        # Update bounds
        bounds = chained_updates(
                self.bounds, self.C_0_nnz, new_objval_at_relaxation=stats["lowerbound"]
                )

        constraints = {
            "min_size": self.min_size,
            "max_size": self.max_size,
            "coef_set": self.coef_set,
            }

        # todo move into factory function
        def rounded_model_size_is_ok(rho):
            zero_idx_rho_ceil = np.equal(np.ceil(rho), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho), 0)
            cannot_round_to_zero = np.logical_not(
                    np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor)
                    )
            rounded_min_coef_size = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            rounded_max_coef_size = np.count_nonzero(rho[self.L0_reg_ind])
            return (
                    rounded_min_coef_size >= self.min_size >= 0
                    and rounded_max_coef_size <= self.max_size
            )

        pool = pool.remove_infeasible(rounded_model_size_is_ok).distinct().sort()

        if len(pool) == 0:
            self.log("all CPA solutions are infeasible")

        # Round CPA solutions
        if settings["use_rounding"] and len(pool) > 0:
            self.log(f"running naive rounding on {len(pool)} solutions")
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))
            rnd_pool, _, _ = round_solution_pool(
                    pool,
                    constraints,
                    max_runtime=settings["rounding_max_runtime"],
                    max_solutions=settings["rounding_max_solutions"],
                    )
            rnd_pool = rnd_pool.compute_objvals(self.get_objval).remove_infeasible(
                    self.is_feasible
                    )
            self.log(f"rounding produced {len(rnd_pool)} integer solutions")
            if len(rnd_pool) > 0:
                pool.append(rnd_pool)
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

        # Sequentially round CPA solutions
        if settings["use_sequential_rounding"] and len(pool) > 0:
            self.log(f"running sequential rounding on {len(pool)} solutions")
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

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
            self.log(
                    f"sequential rounding produced {len(sqrnd_pool)} integer solutions"
                    )

            if len(sqrnd_pool) > 0:
                pool = pool.append(sqrnd_pool)
                self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

        # Polish rounded solutions
        if settings["polishing_after"] and len(pool) > 0:
            self.log("polishing %d solutions" % len(pool))
            self.log("best objective value: %1.4f" % np.min(pool.objvals))
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
                self.log(f"polishing produced {len(dcd_pool)} integer solutions")
                pool.append(dcd_pool)

        # Remove solutions that are not feasible, not integer
        if len(pool) > 0:
            pool = pool.remove_nonintegral().distinct().sort()

        # Update upper and lower bounds
        self.log(f"initialization produced {len(pool)} feasible solutions")

        if len(pool) > 0:
            bounds = chained_updates(
                    bounds, self.C_0_nnz, new_objval_at_feasible=np.min(pool.objvals)
                    )
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

        self.pool.append(pool)

        # Update bounds
        self.bounds = copy(bounds)
        self.has_warmstart = True

    def add_constraint(self, var_names, var_type, values, rhs, sense, name=None):
        """Add a constraint to the MIP.

        Parameters
        ----------
        var_names : list of string
            Variable names to use in constraint.
        var_type : {"rho", "alpha"}
            Either the coefficient (rho) or use term (alpha).
            Alpha is in {0, 1}. Rho is the coefficient being learned.
        values : list of float
            Coefficients to multiply the varible by.
        rhs : float
            Right hand side of the inequality.
        sense : {"G", "L", "E", "R"}
            Must be one of 'G', 'L', 'E', and 'R', indicating greater-than,
            less-than, equality, and ranged constraints, respectively.
        name : str, optional, default: None
            Name of the constraint being added.
            For accessibility from the cplex mip object.

        Notes
        -----
        Formulated as:

            values[0]*variable[0] + ... + values[i]*variable[i] (<, <=, >, >=) rhs,
            where variable[i] is either the coefficient (rho) or use term (alpha).

        """

        # Checks
        if var_type not in ("rho", "alpha"):
            raise ValueError("var_type must be in [\"rho\", \"alpha\"]")

        if len(var_names) != len(values):
            raise ValueError("var_name and values must be the same length.")

        # Anon funcs to get name and index set in mip object (e.g. rho_i, alpha_i)
        get_name = lambda var_name: var_type + '_' + str(self.variable_names.index(var_name))
        get_ind = lambda var_names: [get_name(v) for v in var_names]

        # Add constraint to a queue - it is added to the mip solver add fit time
        self.constraints.append([name, get_ind(var_names), values, sense, rhs])

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
                np.all(self.max_coef >= rho)
                and np.all(self.min_coef <= rho)
                and (self.min_size <= np.count_nonzero(rho[self.L0_reg_ind]) <= self.max_size)
        )

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
            "solution": self.stats.incumbent,
            "objective_value": float("inf"),
            "loss_value":  float("inf"),
            "optimality_gap": float("inf"),
            "run_time": self.stats.total_run_time,
            "solver_time": self.stats.total_solver_time,
            "callback_time": self.stats.total_callback_time,
            "data_time": self.stats.total_data_time,
            "nodes_processed": self.stats.nodes_processed,
            }

        if self.stats.found_solution:
            info.update({
                "objective_value": self.get_objval(self.stats.incumbent),
                "loss_value": self.compute_loss(self.stats.incumbent),
                "optimality_gap": self.stats.relative_gap
                })

        return info


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

    def _init_loss_computation(self):
        """Initalize loss functions."""
        # todo: check if fast/lookup loss is installed
        # todo: refactor loss computation logic to factory functions below
        assert self.loss_computation in ("normal", "weighted", "fast", "lookup")

        use_lookup_table = (
                isinstance(self.coef_set, CoefficientSet) and self._integer_data
        )

        sample_weights = self.sample_weights
        use_weighted = not np.equal(sample_weights, 1.0).all()

        if use_weighted:
            final_loss_computation = "weighted"
        elif use_lookup_table and self.loss_computation == "lookup":
            final_loss_computation = "lookup"
        elif self.loss_computation == "normal":
            final_loss_computation = "normal"
        else:
            final_loss_computation = "fast"

        if final_loss_computation != self.loss_computation:
            warn(f"switching loss computation from {self.loss_computation} to {final_loss_computation}")

        self.loss_computation = final_loss_computation

        if final_loss_computation == "normal":
            import riskslim.loss_functions.log_loss as lf
            self.Z = np.require(self.Z, requirements=["C"])
            self.compute_loss = lambda rho: lf.log_loss_value(self.Z, rho)
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self.Z, rho)
            self.compute_loss_from_scores = (lambda scores: lf.log_loss_value_from_scores(scores))

        elif final_loss_computation == "fast":
            import riskslim.loss_functions.fast_log_loss as lf
            self.Z = np.require(self.Z, requirements=["F"])
            self.compute_loss = lambda rho: lf.log_loss_value(self.Z, np.asfortranarray(rho))
            self.compute_loss_cut = lambda rho: lf.log_loss_value_and_slope(self.Z, np.asfortranarray(rho))
            self.compute_loss_from_scores = (lambda scores: lf.log_loss_value_from_scores(scores))

        elif final_loss_computation == "weighted":
            import riskslim.loss_functions.log_loss_weighted as lf
            self.Z = np.require(self.Z, requirements=["C"])
            total_sample_weights = np.sum(sample_weights)
            self.compute_loss = lambda rho: lf.log_loss_value(self.Z, sample_weights, total_sample_weights, rho)
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
                    max_size=self.max_size,
                    )

            self.log("%d rows in lookup table" % (s_max - s_min + 1))
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

    def _init_coeffs(self):
        """Initialize coefficient constraints.

        Notes
        -----
        If coef_set is passed during initalization, this is ran at initalization time.
        Otherwise, this will be ran when calling the fit method.
        """
        # Coefficients
        self.min_coef = np.array(self.coef_set.lb)
        self.max_coef = np.array(self.coef_set.ub)

        # Vectorized regularization parameters
        self.L0_reg_ind = np.isnan(self.coef_set.c0) + self.coef_set.c0 != 0.0
        self.C_0 = np.array(self.coef_set.c0)

        # Regularization parameter
        if isinstance(self.c0_value, np.ndarray):
            assert np.all(self.c0_value[self.L0_reg_ind] > 0.0), "C0 should be positive"
            self.C_0[self.L0_reg_ind] = self.c0_value[self.L0_reg_ind]
        else:
            assert self.c0_value > 0.0, "C0 should be positive"
            self.C_0[self.L0_reg_ind] = self.c0_value
        self.C_0_nnz = self.C_0[self.L0_reg_ind]

        # Model size constraints
        max_size = np.inf if self.max_size is None else self.max_size
        min_size = 0 if self.min_size is None else self.min_size
        self.max_size = np.minimum(max_size, np.sum(self.L0_reg_ind))
        self.min_size = np.maximum(min_size, 0)

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
                    lb=self.min_coef,
                    ub=self.max_coef,
                    c0=self.c0_value,
                    vtype=self.vtype,
                    print_flag=self.verbose,
                    )
            self._init_coeffs()

        # Check data types and shapes
        check_data(
                self.X, self.y, self.variable_names, self.outcome_name, self.sample_weights
                )

        self.Z = (self.X * self.y).astype(np.float64)

        self._variable_types = np.zeros(self.X.shape[1], dtype="str")
        self._variable_types[:] = "C"
        self._variable_types[np.all(self.X == np.require(self.X, dtype=np.int_), axis=0)] = "I"
        self._variable_types[np.all(self.X == np.require(self.X, dtype=np.bool_), axis=0)] = "B"

        self._integer_data = not np.any(self._variable_types == "C")
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
        self.bounds = Bounds(min_size=self.min_size, max_size=self.max_size)
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
            active_set_flag = self.max_size <= self.n_variables
            polisher = lambda rho: discrete_descent(
                    rho,
                    self.Z,
                    self.C_0,
                    self.max_coef,
                    self.min_coef,
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
        num_max_reg_coefs = self.max_size

        # calculate the smallest and largest score that can be attained by each point
        scores_at_lb = self.Z * self.min_coef
        scores_at_ub = self.Z * self.max_coef
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
