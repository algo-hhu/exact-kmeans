import copy
import json
import logging
import math
import multiprocessing
import multiprocessing.managers
import os
import queue
from itertools import chain, zip_longest
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB
from threadpoolctl import threadpool_limits
from tqdm import tqdm

import exact_kmeans.dynamic_program.dp_plain as dp
import exact_kmeans.init_bounds as init_bounds
from exact_kmeans.util import (
    JsonEncoder,
    compute_centers,
    get_distance,
    kmeans_cost,
    print_variables,
)

# class GurobiFilter(logging.Filter):
#     def __init__(self, name="GurobiFilter"):
#         super().__init__(name)

#     def filter(self, record):
#         return False

# grbfilter = GurobiFilter()

# grblogger = logging.getLogger("gurobipy")
# if grblogger is not None:
#     grblogger.addFilter(grbfilter)
#     grblogger = grblogger.getChild("gurobipy")
#     if grblogger is not None:
#         grblogger.addFilter(grbfilter)

logger = logging.getLogger(__name__)


class ExactKMeans:
    def __init__(
        self,
        n_clusters: int,
        config_file: Union[str, Path] = Path(__file__).parent.resolve()
        / "config"
        / "default.yaml",
        kmeans_iterations: int = 100,
        LB: Optional[list] = None,
        UB: Optional[list] = None,
        outlier: int = 0,
    ) -> None:
        self.k = n_clusters
        self._v = 1
        self._k = self.k + self._v
        self.kmeans_iterations = kmeans_iterations

        self.changed_model_params = {}
        self.changed_bound_model_params = {}
        with Path(config_file).open("r") as f:
            self.config = yaml.safe_load(f)
        for key, value in self.config["model_params"].items():
            self.changed_model_params[key] = value
        for key, value in self.config["bound_model_params"].items():
            self.changed_bound_model_params[key] = value

        self.tolerance_value = 1e-10
        for param in self.changed_model_params:
            if "tol" in param.lower():
                self.tolerance_value = max(
                    self.tolerance_value, self.changed_model_params[param]
                )

        self.constraints = {"bounds": False, "outlier": False}
        self.UB = []
        self.LB = []
        if LB is not None or UB is not None:
            self.constraints["bounds"] = True

            self.LB = LB if LB is not None else [0] * self.k
            self.UB = UB if UB is not None else [np.inf] * self.k

            if len(self.LB) < self.k or len(self.UB) < self.k:
                raise ValueError(
                    f"Length of LB {len(self.LB)} and/or"
                    f"length of UB {len(self.UB)} is not equal n_clusters {self.k}."
                )
            for i in range(self.k):
                if self.LB[i] > self.UB[i]:
                    raise ValueError(
                        f"In position {i}: lower bound is larger than upper bound."
                    )

            logger.info(f"Lower bounds set to {self.LB}, upper bounds set to {self.UB}")

        self.outlier = outlier if outlier is not None else 0
        if self.outlier < 0:
            raise ValueError(f"Number of outliers {self.outlier} is negative.")
        if self.outlier > 0:
            self.constraints["outlier"] = True
            logger.info(f"Number of outliers is set to {self.outlier}")

        logger.info(self.config)
        version = self.config.get("ilp_version", None)
        self.ilp_version = version

        if self.config.get("branch_priorities", False):
            logger.info("Setting variable branch priorities.")
            self.ilp_version += "-priority-x"

        if self.config.get("warm_start", False):
            logger.info("Pre compute initial solution for ILPs.")

        if self.config.get("replace_min", False):
            logger.info("Replacing min function with linear constraints.")
            self.ilp_version += "-replace-min"

        self.ilp_branching_until_level = 0
        if self.config.get("branching_levels", False):
            # This should be at most k - 2
            # At k we already have the final cluster size that we are testing
            # At k - 1 we consider the last cluster size before filling with everything else
            # So we only compute the branching ILP until k - 2
            self.ilp_branching_until_level = min(
                int(self.config.get("branching_levels")), self.k - 2
            )
            logger.info(
                "Computing ILP on branching cluster sizes "
                f"when the cluster sizes are less than {self.ilp_branching_until_level}."
            )
            self.ilp_version += "-branching-ilp"

        if self.config.get("fill_cluster_sizes", False):
            if self.constraints.get("bounds", False) or self.constraints.get(
                "outlier", False
            ):
                logger.info(
                    "Filling of cluster sizes is not suported for k-means with constraints,"
                    "fill_cluster_sizes will be set to False."
                )
                self.config["fill_cluster_sizes"] = False
            else:
                logger.info(
                    "When computing ILP on brnaching cluster sizes "
                    f"fill up cluster sizes to {self.k}."
                )
                self.ilp_version += "-fill-sizes"

        self.num_processes = self.config.get("num_processes", 1)
        if isinstance(self.num_processes, int):
            self.num_processes = min(self.num_processes, multiprocessing.cpu_count())
        elif isinstance(self.num_processes, float):
            self.num_processes = int(self.num_processes * multiprocessing.cpu_count())
        else:
            raise ValueError("num_processes must be an integer or a float.")

        logger.info(
            f"Using {self.num_processes} processes for "
            f"multiprocessing out of {multiprocessing.cpu_count()} total processes."
        )

        self.dp_bounds: Optional[np.ndarray] = None

        self.model: Optional[gp.Model] = None

    def load_run(self, load_existing_run_path: Optional[Path] = None) -> None:
        if load_existing_run_path is not None and load_existing_run_path.exists():
            logger.info(f"Loading existing run from {load_existing_run_path}.")

            with load_existing_run_path.open("r") as f:
                self.existing_run: Dict[str, Any] = json.load(f)
            if "db_bounds" in self.existing_run:
                self.db_bounds = np.array(self.existing_run["dp_bounds"])
            if "cluster_size_objectives" in self.existing_run:
                self.cluster_size_objectives = {
                    int(k): v
                    for k, v in self.existing_run["cluster_size_objectives"].items()
                }
            else:
                self.cluster_size_objectives = {0: 0, 1: 0}

            if "optimal_kmeanspp_cluster_cost" in self.existing_run:
                self.kmeanspp_cluster_cost = self.existing_run[
                    "optimal_kmeanspp_cluster_cost"
                ]
            else:
                self.kmeanspp_cluster_cost = np.inf

            if "processed_cluster_sizes" in self.existing_run:
                self.processed_cluster_sizes = self.existing_run[
                    "processed_cluster_sizes"
                ]
            else:
                self.processed_cluster_sizes = []
        else:
            self.cluster_size_objectives = {0: 0, 1: 0}
            self.kmeanspp_cluster_cost = np.inf
            self.processed_cluster_sizes = []

    def distance_by_index(self, i: int, j: int) -> Any:
        return get_distance(self.X[i - self._v], self.X[j - self._v])

    def change_model_params(
        self, model: gp.Model, bound: bool = False, remove_tolerance: bool = False
    ) -> None:
        params = self.changed_bound_model_params if bound else self.changed_model_params
        for key, value in params.items():
            if remove_tolerance and "tol" in key.lower():
                continue
            model.setParam(key, value)

    #### Variables

    def set_var_x(
        self,
        model: gp.Model,
        variable_type: str = GRB.BINARY,
        different_k: Optional[int] = None,
    ) -> gp.tupledict:
        k = self._k if different_k is None else different_k
        return model.addVars(
            [(i, ll) for i in range(self._v, self._n) for ll in range(self._v, k)],
            vtype=variable_type,
            name="x",
        )

    def set_var_y(
        self,
        model: gp.Model,
        variable_type: str = GRB.BINARY,
        different_k: Optional[int] = None,
    ) -> gp.tupledict:
        k = self._k if different_k is None else different_k
        return model.addVars(
            [
                (i, j, ll)
                for i in range(self._v, self._n - 1)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ],
            vtype=variable_type,
            name="y",
        )

    ####

    def set_constraint_to_lazy(self, constrs: gp.tupledict, info: str) -> None:
        logger.info(f"Setting {info} constraints to lazy.")
        for i in constrs:
            constrs[i].Lazy = 1

    #### Constraints
    def one_cluster_per_point_constraint(
        self, model: gp.Model, x: gp.tupledict, lazy: bool = False, equals: bool = True
    ) -> None:
        if equals:
            constrs = model.addConstrs(
                (x.sum(i, "*") == 1 for i in range(self._v, self._n)),
                name="one_cluster_per_point",
            )
        else:
            constrs = model.addConstrs(
                (x.sum(i, "*") <= 1 for i in range(self._v, self._n)),
                name="one_cluster_per_point",
            )
        if lazy:
            self.set_constraint_to_lazy(constrs, "one cluster per point")

    def fixed_cluster_size_constraint(
        self,
        model: gp.Model,
        x: gp.tupledict,
        cluster_sizes: np.ndarray,
        lazy: bool = False,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        constrs = model.addConstrs(
            (x.sum("*", ll) == cluster_sizes[ll - 1] for ll in range(self._v, k)),
            name="fixed_cluster_size",
        )
        if lazy:
            self.set_constraint_to_lazy(constrs, "fixed cluster size")

    def bound_cluster_size_constraint(
        self,
        model: gp.Model,
        y: gp.tupledict,
        cluster_sizes: np.ndarray,
        lazy: bool = False,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        constrs = model.addConstrs(
            (
                y.sum("*", "*", ll)
                == (cluster_sizes[ll - 1] * (cluster_sizes[ll - 1] - 1)) / 2
                for ll in range(self._v, k)
            ),
            "bound_cluster_sizes",
        )
        if lazy:
            self.set_constraint_to_lazy(constrs, "bound cluster size")

    def bound_cluster_size_constraint_points(
        self,
        model: gp.Model,
        x: gp.tupledict,
        y: gp.tupledict,
        cluster_sizes: np.ndarray,
        lazy: bool = False,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        constrs = model.addConstrs(
            (
                gp.quicksum(y[i, j, ll] for i in range(self._v, j))
                + gp.quicksum(y[j, i, ll] for i in range(j + 1, self._n))
                == (cluster_sizes[ll - 1] - 1) * x[j, ll]
                for j in range(self._v, self._n)
                for ll in range(self._v, k)
            ),
            "bound_cluster_sizes_points",
        )
        if lazy:
            self.set_constraint_to_lazy(constrs, "bound cluster size points")

    def both_points_in_cluster_linear(
        self,
        model: gp.Model,
        x: gp.tupledict,
        y: gp.tupledict,
        lazy: bool = False,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        constrs = model.addConstrs(
            (
                x[i, ll] + x[j, ll] - y[i, j, ll] <= 1
                for i in range(self._v, self._n)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ),
            "both_points_in_cluster_linear",
        )
        if lazy:
            self.set_constraint_to_lazy(constrs, "both_points_in_cluster_linear")

    def upper_bound_both_points_in_cluster(
        self,
        model: gp.Model,
        x: gp.tupledict,
        y: gp.tupledict,
        lazy: bool = False,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        constrs1 = model.addConstrs(
            (
                y[i, j, ll] <= x[i, ll]
                for i in range(self._v, self._n)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ),
            "upper_bound_both_points_in_cluster1",
        )
        constrs2 = model.addConstrs(
            (
                y[i, j, ll] <= x[j, ll]
                for i in range(self._v, self._n)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ),
            "upper_bound_both_points_in_cluster2",
        )
        if lazy:
            self.set_constraint_to_lazy(constrs1, "upper_bound_both_points_in_cluster")
            self.set_constraint_to_lazy(constrs2, "upper_bound_both_points_in_cluster")

    def bound_cost_constraint(
        self,
        model: gp.Model,
        y: gp.tupledict,
        cluster_sizes: np.ndarray,
        kmeanspp_cost: float,
        lazy: bool = False,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        # Add the largest tolerance to the kmeans cost
        bounded_cost = kmeanspp_cost + self.tolerance_value

        constr = model.addConstr(
            gp.quicksum(
                (y[i, j, ll] * self.distance_by_index(i, j))
                / (cluster_sizes[ll - 1] if cluster_sizes[ll - 1] > 0 else 1)
                for i in range(self._v, self._n - 1)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            )
            <= bounded_cost,
            name="bound_cost",
        )
        if lazy:
            logger.info("Setting bound cost constraint to lazy.")
            constr.Lazy = 1

    def both_points_in_cluster_constraint(
        self,
        model: gp.Model,
        x: gp.tupledict,
        y: gp.tupledict,
        different_k: Optional[int] = None,
    ) -> None:
        k = self._k if different_k is None else different_k
        # This constraint cannot be set to lazy because it is non-linear
        model.addConstrs(
            (
                y[i, j, ll] == gp.min_([x[i, ll], x[j, ll]])
                for i in range(self._v, self._n - 1)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ),
            "both_points_in_cluster",
        )

    def get_labels(self) -> np.ndarray:
        assert (
            self.model is not None
        ), "Please run the optimization first to define a model."

        labels = np.zeros(len(self.X), dtype=int)
        self.constraints.get("outlier", False)
        for i in range(self._v, self._n):
            out = True
            for ll in range(self._v, self._k):
                var_by_name = self.model.getVarByName(f"x[{i},{ll}]")
                if var_by_name is None:
                    raise ValueError(f"Variable x[{i},{ll}] not found.")
                if var_by_name.x > 0:  # type: ignore
                    out = False
                    labels[i - self._v] = ll - self._v
            if self.constraints.get("outlier", False) and out is True:
                labels[i - self._v] = self.k
        return labels

    def set_var_branch_priority(
        self, x: gp.tupledict, different_k: Optional[int] = None
    ) -> None:
        k = self._k if different_k is None else different_k
        for i in range(self._v, self._n):
            for ll in range(self._v, k):
                x[i, ll].BranchPriority = 1

    def run_single_cluster_ilp(self, m: int) -> Tuple[int, float]:
        logger.info(f"Running ILP with cluster size bound: {m}")
        start = time()
        model = gp.Model(f"kmeans_bound_{m}")
        self.change_model_params(model, bound=True)
        k = 2
        cluster_sizes = np.array([m])

        x = self.set_var_x(model, variable_type=GRB.BINARY, different_k=k)
        y = self.set_var_y(model, variable_type=GRB.CONTINUOUS, different_k=k)

        for v in y:
            y[v].LB = 0
            y[v].UB = 1

        self.fixed_cluster_size_constraint(model, x, cluster_sizes, different_k=k)

        self.one_cluster_per_point_constraint(model, x, equals=False)

        self.bound_cluster_size_constraint_points(
            model, x, y, cluster_sizes, different_k=k
        )

        self.both_points_in_cluster_linear(model, x, y, different_k=k)
        self.upper_bound_both_points_in_cluster(model, x, y, different_k=k)

        model.setObjective(
            gp.quicksum(
                (y[i, j, ll] * self.distance_by_index(i, j))
                / (cluster_sizes[ll - 1] if cluster_sizes[ll - 1] > 0 else 1)
                for i in range(self._v, self._n - 1)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ),
            GRB.MINIMIZE,
        )

        model.optimize()

        # These models cannot be infeasible, so something is wrong if that happens
        if model.Status == GRB.Status.INFEASIBLE:
            model.computeIIS()
            model.write("cluster_size_model.ilp")
            raise Exception(f"The cluster size model with size {m} is infeasible.")

        logger.info(
            f"Objective for cluster size {m}: {model.ObjVal} "
            f"computed in {time() - start:.3f} seconds."
        )
        obj = model.ObjVal - self.tolerance_value
        del model

        return m, obj

    def run_fixed_cluster_sizes_ilp(
        self,
        cluster_sizes: np.ndarray,
        cost: Optional[float] = None,
        solution: Optional[np.ndarray] = None,
        remove_tolerance: bool = False,
        add_remaining_points: bool = False,
    ) -> gp.Model:
        # If we have all points
        if add_remaining_points:
            # If we don't have all the points and
            # we want to only assign the points fixed by cluster_sizes
            k = len(cluster_sizes) + 1
            equals = False
        else:
            k = self._k
            equals = True

        # TODO: Try reinitializing the same model every time
        model = gp.Model(f"exact_kmeans_{cluster_sizes}")
        self.change_model_params(model, remove_tolerance=remove_tolerance)

        x = self.set_var_x(model, variable_type=GRB.BINARY, different_k=k)
        y = self.set_var_y(model, variable_type=GRB.CONTINUOUS, different_k=k)

        for v in y:
            y[v].LB = 0
            y[v].UB = 1

        if solution is not None:
            for i, label in enumerate(solution):
                if label != k - 1:
                    x[i + self._v, label + self._v].start = 1

        self.one_cluster_per_point_constraint(model, x, equals=equals)

        self.fixed_cluster_size_constraint(model, x, cluster_sizes, different_k=k)

        if self.config.get("replace_min", False):
            self.bound_cluster_size_constraint_points(
                model, x, y, cluster_sizes, different_k=k
            )
            self.both_points_in_cluster_linear(model, x, y, different_k=k)
            self.upper_bound_both_points_in_cluster(model, x, y, different_k=k)
        else:
            self.both_points_in_cluster_constraint(model, x, y, different_k=k)
            self.bound_cluster_size_constraint(model, y, cluster_sizes, different_k=k)

        if cost is not None:
            self.bound_cost_constraint(
                model,
                y,
                cluster_sizes,
                cost,
                different_k=k,
            )

        if self.config.get("branch_priorities", False):
            self.set_var_branch_priority(x, different_k=k)

        model.setObjective(
            gp.quicksum(
                (y[i, j, ll] * self.distance_by_index(i, j))
                / (cluster_sizes[ll - 1] if cluster_sizes[ll - 1] > 0 else 1)
                for i in range(self._v, self._n - 1)
                for j in range(i + 1, self._n)
                for ll in range(self._v, k)
            ),
            GRB.MINIMIZE,
        )

        # presolved_model = model.presolve()
        # presolved_model.printStats()
        # presolved_model.write("presolved.lp")
        # presolved_model.write("presolved.mps")

        model.optimize()

        # model.computeIIS()
        # model.write("model.lp")

        # model.write("model.lp")
        # model.write("model.mps")

        return model

    def get_fixed_cluster_sizes_ilp_result(
        self,
        cluster_sizes: np.ndarray,
        tightest_upper_bound: float,
        add_remaining_points: bool = False,
    ) -> Tuple[Union[str, float], float]:
        objective_value: Union[str, float] = "infeasible"
        logger.info(f"Running ILP with cluster sizes: {cluster_sizes}")
        start_sol = None
        if self.config.get("warm_start", False):
            logger.info(
                f"Compute intial solution for ILP with cluster sizes {cluster_sizes}"
                f"and outliers {self.n-sum(cluster_sizes)}."
            )
            ILP_init = init_bounds.KMeans_bounded(
                n_clusters=len(cluster_sizes),
                LB=list(cluster_sizes),
                UB=list(cluster_sizes),
                outlier=self.n - sum(cluster_sizes),
                kmeans_iterations=10,
            )
            # need this, otherwise scikit learn KMeans does not work within multiprocessing
            with threadpool_limits(user_api="openmp", limits=1):
                ILP_init.fit(self.X)
                logger.info(
                    f"Found solution with cluster sizes {cluster_sizes} "
                    f"and outliers {self.n-sum(cluster_sizes)} "
                    f"with cost {ILP_init.best_inertia}."
                )
            if ILP_init.best_inertia <= tightest_upper_bound:
                start_sol = ILP_init.best_labels
            else:
                logger.info(
                    f"Computed intial solution for ILP with cost {ILP_init.best_inertia} "
                    f"exceeds currently best upper bound {tightest_upper_bound}. "
                    "Warm start not possible."
                )
        start = time()
        model = self.run_fixed_cluster_sizes_ilp(
            cluster_sizes=cluster_sizes,
            cost=tightest_upper_bound,
            solution=start_sol,
            add_remaining_points=add_remaining_points,
        )
        ILP_time = time() - start
        logger.info(
            f"Model with cluster sizes {cluster_sizes} took " f"{ILP_time:.3f} seconds."
        )

        if model.Status == GRB.Status.INF_OR_UNBD:
            raise ValueError(
                f"Model with cluster sizes {cluster_sizes} "
                "is infeasible or unbounded."
            )
        elif model.Status == GRB.Status.INFEASIBLE:
            logger.info(f"Cluster sizes {cluster_sizes} are infeasible, skipping...")
        else:
            logger.info(
                f"A model with cluster sizes {cluster_sizes} is "
                f"feasible with objective {model.ObjVal}."
            )
            objective_value = model.ObjVal

        return objective_value, ILP_time

    def fix_rem_cluster_sizes(self, cluster_sizes: List) -> List:
        new_cluster_sizes = copy.deepcopy(cluster_sizes)

        if len(new_cluster_sizes) == self.k:
            return new_cluster_sizes

        n_fixed_points = sum(new_cluster_sizes)
        search_start = max(
            1,
            math.ceil(
                (self.n - n_fixed_points - self.outlier)
                / (self.k - len(new_cluster_sizes))
            ),
        )
        if len(new_cluster_sizes) > 0:
            search_end = min(
                new_cluster_sizes[-1],
                self.n - n_fixed_points - self.k + len(new_cluster_sizes) + 1,
            )

        else:
            search_end = self.n - n_fixed_points - self.k + len(new_cluster_sizes) + 1

        new_cluster_sizes.append(search_start)
        n_remaining_points = self.n - n_fixed_points - search_end

        while len(new_cluster_sizes) < self.k:
            search_start = max(
                1,
                math.ceil(
                    (n_remaining_points - self.outlier)
                    / (self.k - len(new_cluster_sizes))
                ),
            )
            search_end_new = min(
                search_end, n_remaining_points - self.k + len(new_cluster_sizes) + 1
            )
            search_end = search_end_new
            n_remaining_points -= search_end
            new_cluster_sizes.append(search_start)

        assert (
            sum(new_cluster_sizes) <= self.n
        ), "fix_rem_cluster_sizes: sum new_cluster_sizes exceeds number of points"

        return new_cluster_sizes

    def check_if_processed(
        self, current_cluster_sizes: np.ndarray
    ) -> Tuple[Optional[str], bool]:
        for obj, _, sizes in self.processed_cluster_sizes:
            if sizes == current_cluster_sizes:
                logger.info(
                    f"Cluster sizes {current_cluster_sizes} have "
                    "already been previously processed, skipping..."
                )
                return (obj, True)
        return (None, False)

    def enumerate_sizes(
        self,
        task_queue: queue.Queue,
        output_list: Any,
        tightest_upper_bound: multiprocessing.managers.ValueProxy,
        lock: Any,  # multiprocessing.managers.AcquirerProxy,
    ) -> None:
        if self.constraints.get("bounds", False):
            bounded = init_bounds.KMeans_bounded(
                n_clusters=self.k, LB=self.LB, UB=self.UB
            )

        while True:
            try:
                current_cluster_sizes = task_queue.get(
                    timeout=1  # Timeout to allow graceful exit
                )
            except queue.Empty:
                break

            already_processed = False
            obj, already_processed = self.check_if_processed(current_cluster_sizes)
            if already_processed:
                if obj == "branch":
                    remaining_points = self.n - sum(current_cluster_sizes)
                    search_start = max(
                        1,
                        math.ceil(
                            (remaining_points - self.outlier)
                            / (self.k - len(current_cluster_sizes))
                        ),
                    )
                    search_end = min(
                        current_cluster_sizes[-1],
                        remaining_points - self.k + len(current_cluster_sizes) + 1,
                    )
                    logger.info(
                        "Find next position in cluster sizes:"
                        f"[{search_start}, {search_end}]."
                    )
                    for m in range(search_start, search_end + 1):
                        logger.info(
                            f"Enumerating cluster sizes: {current_cluster_sizes + [m]}"
                        )
                        task_queue.put(current_cluster_sizes + [m])
                continue

            logger.info(f"Current cluster sizes: {current_cluster_sizes}.")
            n_fixed_points = sum(current_cluster_sizes)
            k_fixed = len(current_cluster_sizes)

            # Lower bound on the cost of a clustering with cluster_sizes as constraint,
            # We use the results from the DP to find a better lower bound

            assert self.dp_bounds is not None, "DP bounds have not been computed."
            points_left = max(self.k - k_fixed, self.n - n_fixed_points - self.outlier)
            sum_bound = (
                sum(self.cluster_size_objectives[m] for m in current_cluster_sizes)
                + self.dp_bounds[points_left][self.k - k_fixed]
            )

            test_sizes = current_cluster_sizes
            ILP_time = 0.0
            found_bound: Union[str, float]

            if self.constraints.get("bounds", False):
                _, proceed = bounded.check_bound_feasibility(
                    {i: test_sizes[i] for i in range(len(test_sizes))}
                )
                if proceed is False:
                    found_bound = "constr_infeasible"
                    logger.info(
                        "Cluster sizes are infeasible for desired "
                        "constraints, skipping..."
                    )
                    with lock:
                        output_list.append(
                            (found_bound, ILP_time, current_cluster_sizes)
                        )
                    continue

            # If the sum of our current bounds is greater than the upper bound, we can skip
            if sum_bound > tightest_upper_bound.value:
                found_bound = "sum_bound_greater"
                logger.info(
                    f"Lower bound {sum_bound} is greater than the "
                    f"current upper bound {tightest_upper_bound.value}, skipping..."
                )

            # If we have the same number of cluster sizes as the number of clusters
            if len(current_cluster_sizes) == self.k:
                add_out = self.constraints.get("outlier", False)
                found_bound, ILP_time = self.get_fixed_cluster_sizes_ilp_result(
                    current_cluster_sizes,
                    tightest_upper_bound.value,
                    add_remaining_points=add_out,
                )
                if (
                    isinstance(found_bound, float)
                    and found_bound < tightest_upper_bound.value
                ):
                    logger.info(
                        "Found a better upper bound: "
                        f"{found_bound} < {tightest_upper_bound.value}."
                    )
                    with lock:
                        tightest_upper_bound.value = found_bound
            else:
                found_bound = "branch"
                remaining_points = self.n - sum(current_cluster_sizes)
                search_start = max(
                    1,
                    math.ceil(
                        (remaining_points - self.outlier)
                        / (self.k - len(current_cluster_sizes))
                    ),
                )
                search_end = min(
                    current_cluster_sizes[-1],
                    remaining_points - self.k + len(current_cluster_sizes) + 1,
                )
                # Run the ILP if we have more than one cluster size to
                # see if we should branch from here
                # It does not make sense to run it if we only have one value
                # Because we have done it before with the other ILP
                if (
                    len(current_cluster_sizes) >= 1
                    and len(current_cluster_sizes) <= self.ilp_branching_until_level
                ):
                    if self.config.get("fill_cluster_sizes", False):
                        test_sizes = self.fix_rem_cluster_sizes(current_cluster_sizes)
                    else:
                        test_sizes = current_cluster_sizes + [search_start]
                    logger.info(
                        f"Current cluster sizes: "
                        f"{current_cluster_sizes} replaced by {test_sizes}"
                    )
                    sum_bound = sum(self.cluster_size_objectives[m] for m in test_sizes)
                    if sum_bound > tightest_upper_bound.value:
                        found_bound = "sum_bound_greater"
                        logger.info(
                            f"Lower bound {sum_bound} is greater than the "
                            f"current upper bound {tightest_upper_bound.value}, skipping..."
                        )
                    else:
                        (
                            found_bound,
                            ILP_time,
                        ) = self.get_fixed_cluster_sizes_ilp_result(
                            test_sizes,
                            tightest_upper_bound.value,
                            add_remaining_points=True,
                        )

                    if not self.config.get("fill_cluster_sizes", False) and isinstance(
                        found_bound, float
                    ):
                        n_fixed_points += search_end
                        k_fixed += 1
                        dp_bound = (
                            found_bound
                            + self.dp_bounds[self.n - n_fixed_points][self.k - k_fixed]
                        )
                        logger.info(
                            f"Bound for {test_sizes} ({found_bound}) with DP bound ({dp_bound})"
                        )
                        if (
                            np.isfinite(dp_bound).all()
                            and dp_bound > tightest_upper_bound.value
                        ):
                            logger.info(
                                f"Bound for {test_sizes} ({found_bound}) "
                                f"with DP bound ({dp_bound}) "
                                "is greater than the current upper bound "
                                f"{tightest_upper_bound.value}, skipping..."
                            )
                            found_bound = "ilp_sum_bound_greater"
                if found_bound not in {
                    "infeasible",
                    "ilp_sum_bound_greater",
                    "constr_infeasible",
                }:
                    found_bound = "branch"
                    # If the program is feasible and we have less than k clusters
                    # we need to select the next cluster size,
                    # and that shouldn't be larger than the largest size
                    # and should also keep into account how many points still exist

                    logger.info(
                        "Find next position in cluster sizes:"
                        f"[{search_start}, {search_end}]."
                    )
                    for m in range(search_start, search_end + 1):
                        logger.info(
                            f"Enumerating cluster sizes: {current_cluster_sizes + [m]}"
                        )
                        task_queue.put(current_cluster_sizes + [m])

            with lock:
                output_list.append((found_bound, ILP_time, current_cluster_sizes))

    def compute_cluster_size_objectives(self) -> None:
        # If the cluster sizes have not already been computed
        # Iterate through all the possible cluster sizes to find
        # the largest size that makes sense
        start = time()

        # in case we have upper and lower bounds:
        # cluster sizes must lie between 1 and the highest upper bound
        start_bound = max(self.cluster_size_objectives.keys()) + 1
        if self.constraints.get("bounds", False):
            end_bound = min(self.n + 1, max(self.UB) + 1)
        else:
            end_bound = self.n + 1

        greater_string = (
            "Bound {objval} for cluster size {m} is greater than kmeans cost "
            f"{self.kmeanspp_cluster_cost}, stopping..."
        )

        # This is for ease of understanding if the problem is with multiprocessing
        # or with gurobi when the program does not run
        if self.num_processes == 1:
            for i in range(start_bound, end_bound):
                m, objval = self.run_single_cluster_ilp(i)
                if objval > self.kmeanspp_cluster_cost:
                    logger.info(greater_string.format(objval=objval, m=m))
                    break
                self.cluster_size_objectives[m] = objval
        else:
            with multiprocessing.Pool(processes=self.num_processes) as pool:
                try:
                    for m, objval in tqdm(
                        pool.imap(
                            self.run_single_cluster_ilp,
                            range(start_bound, end_bound),
                        ),
                        total=end_bound - start_bound,
                    ):
                        # If we ever get a larger cost than kmeans, we can stop
                        if objval > self.kmeanspp_cluster_cost:
                            logger.info(greater_string.format(objval=objval, m=m))
                            pool.terminate()
                            break
                        self.cluster_size_objectives[m] = objval
                except KeyboardInterrupt:
                    logger.info("Received KeyboardInterrupt, stopping the pool.")
                    pool.terminate()
                    raise KeyboardInterrupt

        logger.info(
            f"Lower bound computation for cluster sizes took {time() - start:.3f} seconds."
        )

        for m in sorted(self.cluster_size_objectives.keys()):
            logger.info(
                f"Bound for cluster size {m}: {self.cluster_size_objectives[m]}"
            )

    def compute_best_cluster_sizes(
        self, kmeanspp_sizes: List[int]
    ) -> Tuple[np.ndarray, float]:

        # when lower- and upper bounds are provided:
        # largest cluster size has to be between
        # lower and upper bound of bounds pair with highest upper bound
        # outliers do not influence this

        m_max = max(self.cluster_size_objectives.keys())
        m_min = math.ceil((self.n - self.outlier) / self.k)

        if self.constraints.get("bounds", False):
            ub_max = max(self.UB)
            indices = []
            for i in range(self.k):
                if self.UB[i] == ub_max:
                    indices += [i]
            lb_min = 0
            for i in indices:
                if self.LB[i] > lb_min:
                    lb_min = self.LB[i]

            m_min = max(m_min, lb_min)
            m_max = min(m_max, ub_max)

        logger.info(
            f"Iterate through all possible maximum cluster sizes: [{m_max}, {m_min}]."
        )

        start = time()
        # Create a manager to handle shared objects
        manager = multiprocessing.Manager()
        # Create shared variables for the return values
        best_obj = manager.Value("d", self.kmeanspp_cluster_cost)
        # create a list to store the processed cluster sizes
        output_list = manager.list()
        # Create a lock for synchronizing access to the shared value
        lock = manager.Lock()

        task_queue: multiprocessing.Queue = multiprocessing.Queue()

        # First put the biggest size of kmeans++ in the queue
        m_max_kmeanspp = kmeanspp_sizes[0]
        task_queue.put([m_max_kmeanspp])

        smaller_ms = [m for m in range(m_max_kmeanspp - 1, m_min - 1, -1)]
        larger_ms = [m for m in range(m_max_kmeanspp + 1, m_max + 1)]

        # Interleave the smaller and larger cluster sizes
        for m in chain.from_iterable(zip_longest(smaller_ms, larger_ms)):
            # If the two lists have different sizes some of the values are None
            if m is None:
                continue
            task_queue.put([m])

        # Create a pool of worker processes
        try:
            processes = []
            for _ in range(self.num_processes):
                p = multiprocessing.Process(
                    target=self.enumerate_sizes,
                    args=(task_queue, output_list, best_obj, lock),
                )
                p.start()
                processes.append(p)

            # Wait for all worker processes to finish
            for p in processes:
                p.join()

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt, stopping each process.")
            for p in processes:
                p.terminate()

            self.processed_cluster_sizes = list(output_list)

            raise KeyboardInterrupt

        logger.info(
            f"Branch&Bound for cluster sizes took {time() - start:.3f} seconds with "
            f"a final objective of {best_obj.value}."
        )

        self.processed_cluster_sizes = list(output_list)
        logger.info(f"number of processed sizes: {len(self.processed_cluster_sizes)}")
        best_tmp_obj: Optional[float] = None
        best_sizes: Optional[np.ndarray] = None

        for obj, _, sizes in self.processed_cluster_sizes:
            if isinstance(obj, float) and (best_tmp_obj is None or obj < best_tmp_obj):
                best_tmp_obj = obj
                best_sizes = sizes

        assert (
            best_sizes is not None and best_tmp_obj is not None
        ), f"No feasible solution was found during Branch&Bound: {self.processed_cluster_sizes}"

        if not np.isclose(best_tmp_obj, best_obj.value):
            logger.error(
                f"Best objective found during Branch&Bound {best_tmp_obj} "
                "does not match objective found in run without "
                f"tolerance {best_obj.value}."
            )
        return np.array(best_sizes), best_obj.value

    def sort_labels(
        self, kmeanspp_labels: np.ndarray, out_label: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        cluster_labels, cluster_sizes = np.unique(kmeanspp_labels, return_counts=True)
        if out_label is not None:
            for i, label in enumerate(cluster_labels):
                if label == out_label:
                    cluster_labels = np.delete(cluster_labels, i)
                    cluster_sizes = np.delete(cluster_sizes, i)
                    break

        sorted_sizes = sorted(cluster_sizes, reverse=True)
        logger.info(f"KMeans++ cluster sizes: {sorted_sizes}")

        sorted_map = {v: i for i, v in enumerate(np.argsort(-cluster_sizes))}

        initial_labels = [sorted_map[ll] for ll in kmeanspp_labels if ll != out_label]

        return initial_labels, sorted_sizes

    def compute_initial_cost_bound(self) -> Tuple[float, np.ndarray]:

        if self.constraints.get("bounds", False):
            logger.info(f"Lower bounds {self.LB} and upper bounds {self.UB} provided. ")
            if self.constraints.get("outlier", False):
                logger.info(f"Number of outliers is at most {self.outlier}.")
                kmeans_init_b = init_bounds.KMeans_bounded(
                    self.k, self.kmeans_iterations, self.LB, self.UB, self.outlier
                )

            else:
                kmeans_init_b = init_bounds.KMeans_bounded(
                    self.k,
                    self.kmeans_iterations,
                    self.LB,
                    self.UB,
                )
            kmeans_init_b.fit(self.X)
            best_inertia = kmeans_init_b.best_inertia
            best_labels = kmeans_init_b.best_labels

        elif self.constraints.get("outlier", False):
            logger.info(f"Number of outliers is at most {self.outlier}.")
            kmeans_init_o = init_bounds.KMeans_outlier(
                self.k, self.kmeans_iterations, self.outlier
            )
            kmeans_init_o.fit(self.X)
            best_inertia = kmeans_init_o.best_inertia
            best_labels = kmeans_init_o.best_labels

        else:
            logger.info("No bounds provided, compute vanilla kmeans++ solution.")
            kmeans_init = init_bounds.KMeans_vanilla(
                self.k,
                self.kmeans_iterations,
            )
            kmeans_init.fit(self.X)
            best_inertia = kmeans_init.best_inertia
            best_labels = kmeans_init.best_labels

        return best_inertia, best_labels

    def get_ilp_result_without_tolerance(
        self, best_sizes: np.ndarray
    ) -> Tuple[float, gp.Model]:
        start_sol = None
        if self.config.get("warm_start", False):
            logger.info(
                f"Compute intial solution for ILP with cluster sizes {best_sizes}"
                f"and outliers {self.n-sum(best_sizes)}."
            )
            ILP_init = init_bounds.KMeans_bounded(
                n_clusters=len(best_sizes),
                LB=list(best_sizes),
                UB=list(best_sizes),
                outlier=self.n - sum(best_sizes),
                kmeans_iterations=10,
            )
            ILP_init.fit(self.X)
            logger.info(
                f"Found solution with cluster sizes {best_sizes} and outliers"
                f"{self.n-sum(best_sizes)} with cost {ILP_init.best_inertia}."
            )
            start_sol = ILP_init.best_labels

        start = time()
        ILP_model = self.run_fixed_cluster_sizes_ilp(
            cluster_sizes=best_sizes,
            cost=None,
            solution=start_sol,
            remove_tolerance=True,
            add_remaining_points=self.constraints.get("outlier", False),
        )
        ILP_time = time() - start
        logger.info(f"Final ILP took {ILP_time:.3f} seconds.")
        return (ILP_time, ILP_model)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Any = None,
        sample_weight: Optional[Sequence[float]] = None,
        kmeanspp_labels: Optional[np.ndarray] = None,
        load_existing_run_path: Optional[Path] = None,
        cache_current_run_path: Optional[Path] = None,
    ) -> "ExactKMeans":

        if isinstance(X, pd.DataFrame):
            self.X = X.values
        elif isinstance(X, np.ndarray):
            self.X = X
        else:
            raise ValueError("Please convert the input data to a numpy array.")
        self.n = len(X)
        self._n = self.n + self._v

        if self.constraints.get("bounds", False):
            if sum(self.LB) > self.n:
                raise ValueError(
                    f"Sum of lower bounds {sum(self.LB)} exceeds number of points {self.n}."
                )
            if sum(self.UB) + self.outlier < self.n:
                raise ValueError(
                    f"Sum of upper bounds {sum(self.UB)}  and outliers {self.outlier}"
                    f"is smaller than number of points {self.n}."
                )
        if self.constraints.get("outlier", False) and self.outlier >= self.n:
            raise ValueError(
                f"Number of outliers {self.outlier}"
                f"is greater equal number of points {self.n}."
            )
        self.initial_labels = None
        self.load_run(load_existing_run_path)
        self.cache_current_run_path = cache_current_run_path

        if self.constraints.get("bounds", False) and (
            sum(self.LB) + self.outlier == self.n
            or sum(self.UB) + self.outlier == self.n
        ):
            best_sizes = self.LB if (sum(self.LB) + self.outlier == self.n) else self.UB
            logger.info(
                f"Only cluster sizes {best_sizes} and number of outliers"
                f" {self.outlier} possible. Skipping branch and bound."
            )
            logger.info(f"Running ILP with best cluster sizes: {best_sizes}.")
            ILP_time, self.model = self.get_ilp_result_without_tolerance(best_sizes)
            self.processed_cluster_sizes = [(self.model.ObjVal, ILP_time, best_sizes)]
            logger.info(
                f"The best found objective was {self.model.ObjVal} with size {best_sizes}. "
            )
        else:
            kmeanspp_cost = np.inf
            if kmeanspp_labels is None:
                kmeanspp_cost, kmeanspp_labels = self.compute_initial_cost_bound()
            else:
                # if the solution constains outliers,
                # they have label k and are ignored in the cost computation
                kmeanspp_cost = kmeans_cost(kmeanspp_labels, points=self.X, k=self.k)

            assert (
                kmeanspp_labels is not None
            ), "KMeans++ labels must be either provided or computed before continuing."

            logger.info("Chosen initial KMeans++ solution with cost: %f", kmeanspp_cost)

            try:
                # if the solution constains outliers, they have label k and will be removed
                self.initial_labels, kmeanspp_sizes = self.sort_labels(
                    kmeanspp_labels, out_label=self.k
                )
                self.kmeanspp_cluster_cost = kmeanspp_cost

                self.compute_cluster_size_objectives()

                # Construct lower bounds for clustering sizes using the dynamic program
                if self.dp_bounds is None or self.dp_bounds.sum() == 0:
                    self.dp_bounds = dp.compute_bounds(
                        self.n, self.k, self.cluster_size_objectives
                    )

                best_sizes, best_obj = self.compute_best_cluster_sizes(kmeanspp_sizes)
            except KeyboardInterrupt:
                store_path = (
                    Path(f"exact_kmeans_pid_{os.getpid()}.json")
                    if self.cache_current_run_path is None
                    else self.cache_current_run_path
                )
                logger.info(
                    "Received KeyboardInterrupt, stopping optimization "
                    f"and storing to {store_path}."
                )

                existing_run = {
                    "dp_bounds": self.dp_bounds.tolist()
                    if self.dp_bounds is not None
                    else [],
                    "cluster_size_objectives": self.cluster_size_objectives,
                    "optimal_kmeanspp_cluster_cost": self.kmeanspp_cluster_cost,
                    "processed_cluster_sizes": self.processed_cluster_sizes,
                }
                with store_path.open("w") as f:
                    json.dump(existing_run, f, cls=JsonEncoder)

                exit(0)

            logger.info(
                f"Re-running ILP with best cluster sizes: {best_sizes} and cost {best_obj}."
            )

            _, self.model = self.get_ilp_result_without_tolerance(best_sizes)

            if not np.isclose(self.model.ObjVal, best_obj, atol=1e-04):
                logger.error(
                    f"Objective value of final model {self.model.ObjVal} "
                    f"does not match best objective value {best_obj}."
                )
            logger.info(
                f"The best found objective was {self.model.ObjVal} with size {best_sizes} "
                f"compared to initial bound {self.kmeanspp_cluster_cost}."
            )

        self.labels_ = self.get_labels()
        self.cluster_centers_ = compute_centers(self.X, self.labels_)
        self.inertia_ = self.model.ObjVal
        self.best_cluster_sizes = best_sizes

        return self

    def print_model(self, results_folder: Path, result_name: str) -> None:
        assert (
            self.model is not None
        ), "Please run the optimization first to define a model."
        self.model.write(str(results_folder / f"{result_name}.mps"))

        with open(results_folder / f"{result_name}.txt", "w") as of:
            vars = self.model.getVars()
            vals = [v.x for v in vars]  # type: ignore
            for txt in print_variables(vars, vals):
                of.write(txt)
