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
from typing import Any, Dict, List, Optional, Tuple, Union

import gurobipy as gp
import numpy as np
import pandas as pd
import yaml
from gurobipy import GRB
from sklearn.cluster import KMeans
from tqdm import tqdm

import exact_kmeans.dynamic_program.dp_plain as dp
from exact_kmeans.util import JsonEncoder, get_distance, kmeans_cost, print_variables

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
        X: Union[np.ndarray, pd.DataFrame],
        k: int,
        config_file: Union[str, Path] = "config/default.yaml",
        cache_current_run_path: Optional[Path] = None,
        load_existing_run_path: Optional[Path] = None,
        kmeans_iterations: int = 100,
    ) -> None:
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        elif isinstance(X, np.ndarray):
            self.X = X
        else:
            raise ValueError("Please convert the input data to a numpy array.")
        self.k = k
        self.n = len(X)

        # self.dmax = compute_largest_distance(X)

        self._v = 1
        self._n = self.n + self._v
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

        self.tolerance_value = 0
        for param in self.changed_model_params:
            if "tol" in param.lower():
                self.tolerance_value = max(
                    self.tolerance_value, self.changed_model_params[param]
                )

        logger.info(self.config)
        version = self.config.get("ilp_version", None)
        self.ilp_version = version

        if self.config.get("branch_priorities", False):
            logger.info("Setting variable branch priorities.")
            self.ilp_version += "-priority-x"

        if self.config.get("replace_min", False):
            logger.info("Replacing min function with linear constraints.")
            self.ilp_version += "-replace-min"

        self.ilp_branching_until_level = 0
        if self.config.get("branching_levels", False):
            self.ilp_branching_until_level = int(self.config.get("branching_levels"))
            logger.info(
                "Computing ILP on branching cluster sizes "
                f"when the cluster sizes are less than {self.ilp_branching_until_level}."
            )
            self.ilp_version += "-branching-ilp"

        if self.config.get("fill_cluster_sizes", False):
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

        self.cache_current_run_path = cache_current_run_path

        if load_existing_run_path is not None and load_existing_run_path.exists():
            logger.info(f"Loading existing run from {load_existing_run_path}.")

            with load_existing_run_path.open("r") as f:
                self.existing_run: Dict[str, Any] = json.load(f)
            if "db_bounds" in self.existing_run:
                self.db_bounds = np.array(self.existing_run["dp_bounds"])
            else:
                self.dp_bounds = np.zeros((self.n + 1, self.k + 1))
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
            self.dp_bounds = np.zeros((self.n + 1, self.k + 1))
            self.cluster_size_objectives = {0: 0, 1: 0}
            self.kmeanspp_cluster_cost = np.inf
            self.processed_cluster_sizes = []

        self.model: Optional[gp.Model] = None

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

        # # Add an x such that Cost <= C + x
        # if self.config.get("bound_cost_dmax", False):
        #     bounded_cost += self.dmax

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
        for i in range(self._v, self._n):
            for ll in range(self._v, self._k):
                var_by_name = self.model.getVarByName(f"x[{i},{ll}]")
                if var_by_name is None:
                    raise ValueError(f"Variable x[{i},{ll}] not found.")
                if var_by_name.x > 0:  # type: ignore
                    labels[i - self._v] = ll - self._v
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

        # if self.config.get("branch_priorities", False):
        # self.set_var_branch_priority(x, different_k=k)

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
        obj = model.ObjVal
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
        start = time()
        model = self.run_fixed_cluster_sizes_ilp(
            cluster_sizes=cluster_sizes,
            cost=tightest_upper_bound,
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
        new_cluster_sizes = cluster_sizes

        if len(new_cluster_sizes) == self.k:
            return new_cluster_sizes

        n_fixed_points = sum(new_cluster_sizes)
        search_start = math.ceil(
            (self.n - n_fixed_points) / (self.k - len(new_cluster_sizes))
        )
        if len(new_cluster_sizes) > 0:
            search_end = min(
                new_cluster_sizes[-1],
                self.n - n_fixed_points - self.k + len(new_cluster_sizes) + 1,
            )

        else:
            search_end = self.n - n_fixed_points - self.k + len(new_cluster_sizes) + 1

        new_cluster_sizes = new_cluster_sizes + [search_start]
        n_remaining_points = self.n - n_fixed_points - search_end

        while len(new_cluster_sizes) < self.k:
            search_start = math.ceil(
                n_remaining_points / (self.k - len(new_cluster_sizes))
            )
            search_end_new = min(
                search_end, n_remaining_points - self.k + len(new_cluster_sizes) + 1
            )
            search_end = search_end_new
            n_remaining_points -= search_end
            new_cluster_sizes = new_cluster_sizes + [search_start]

        assert (
            sum(new_cluster_sizes) <= self.n
        ), "fix_rem_cluster_sizes: sum new_cluster_sizes exceeds number of points"

        return new_cluster_sizes

    def enumerate_sizes(
        self,
        task_queue: queue.Queue,
        output_queue: queue.Queue,
        tightest_upper_bound: multiprocessing.managers.ValueProxy,
        lock: Any,  # multiprocessing.managers.AcquirerProxy,
    ) -> None:
        while True:
            try:
                current_cluster_sizes = task_queue.get(
                    timeout=1  # Timeout to allow graceful exit
                )
            except queue.Empty:
                break

            for obj, t, sizes in self.processed_cluster_sizes:
                if sizes == current_cluster_sizes:
                    logger.info(
                        f"Cluster sizes {current_cluster_sizes} have "
                        "already been previously processed, skipping..."
                    )
                    output_queue.put((obj, t, sizes))
                    return
            logger.info(f"Current cluster sizes: {current_cluster_sizes}.")
            n_fixed_points = sum(current_cluster_sizes)
            k_fixed = len(current_cluster_sizes)

            # Lower bound on the cost of a clustering with cluster_sizes as constraint,
            # We use the results from the DP to find a better lower bound
            sum_bound = (
                sum(self.cluster_size_objectives[m] for m in current_cluster_sizes)
                + self.dp_bounds[self.n - n_fixed_points][self.k - k_fixed]
            )

            test_sizes = current_cluster_sizes
            ILP_time = 0.0

            # If the sum of our current bounds is greater than the upper bound, we can skip
            if sum_bound > tightest_upper_bound.value:
                found_bound: Union[str, float] = "sum_bound_greater"
                logger.info(
                    f"Upper bound {sum_bound} is greater than the "
                    f"current upper bound {tightest_upper_bound.value}, skipping..."
                )
            # If we have the same number of cluster sizes as the number of clusters
            elif len(current_cluster_sizes) == self.k:
                found_bound, ILP_time = self.get_fixed_cluster_sizes_ilp_result(
                    current_cluster_sizes, tightest_upper_bound.value
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
                remaining_points = self.n - n_fixed_points
                search_start = math.ceil(
                    remaining_points / (self.k - len(current_cluster_sizes))
                )
                search_end = min(
                    current_cluster_sizes[-1],
                    remaining_points - self.k + len(current_cluster_sizes) + 1,
                )
                # Run the ILP if we have more than one cluster size to
                # see if we should branch from here
                if len(current_cluster_sizes) <= self.ilp_branching_until_level:
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
                        found_bound, ILP_time = self.get_fixed_cluster_sizes_ilp_result(
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
                        if dp_bound > tightest_upper_bound.value:
                            logger.info(
                                f"Bound for {test_sizes} ({found_bound}) "
                                f"with DP bound ({dp_bound}) "
                                "is greater than the current upper bound "
                                f"{tightest_upper_bound.value}, skipping..."
                            )
                            found_bound = "ilp_sum_bound_greater"
                if found_bound not in {"infeasible", "ilp_sum_bound_greater"}:
                    found_bound = "branch"
                    # If the program is feasible and we have less than k clusters
                    # we need to select the next cluster size,
                    # and that shouldn't be larger than the largest size
                    # and should also keep into account how many points still exist

                    logger.info(
                        f"Find next position in cluster sizes: [{search_start}, {search_end}]."
                    )
                    for m in range(search_start, search_end + 1):
                        logger.info(
                            f"Enumerating cluster sizes: {current_cluster_sizes + [m]}"
                        )
                        task_queue.put(current_cluster_sizes + [m])

            output_queue.put((found_bound, ILP_time, current_cluster_sizes))

    def compute_cluster_size_objectives(self) -> None:
        # If the cluster sizes have not already been computed
        # Iterate through all the possible cluster sizes to find
        # the largest size that makes sense
        start = time()
        start_bound = max(self.cluster_size_objectives.keys()) + 1

        greater_string = (
            "Bound {objval} for cluster size {m} is greater than kmeans cost "
            f"{self.kmeanspp_cluster_cost}, stopping..."
        )

        # This is for ease of understanding if the problem is with multiprocessing
        # or with gurobi when the program does not run
        if self.num_processes == 1:
            for i in range(start_bound, self.n + 1):
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
                            range(start_bound, self.n + 1),
                        ),
                        total=self.n - start_bound + 1,
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
        m_max = max(self.cluster_size_objectives.keys())
        m_min = math.ceil(self.n / self.k)

        logger.info(
            f"Iterate through all possible maximum cluster sizes: [{m_max}, {m_min}]."
        )

        start = time()
        # Create a manager to handle shared objects
        manager = multiprocessing.Manager()
        # Create shared variables for the return values
        best_obj = manager.Value("d", self.kmeanspp_cluster_cost)

        # Create a lock for synchronizing access to the shared value
        lock = manager.Lock()

        # Get the largest cluster size that is less than the kmeans cost
        # for m in range(m_max, m_min - 1, -1):
        #     obj = self.enumerate_sizes([m], upper_bound=best_obj)
        #     best_obj = min(best_obj, obj)
        task_queue: multiprocessing.Queue = multiprocessing.Queue()
        output_queue: multiprocessing.Queue = multiprocessing.Queue()

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
                    args=(task_queue, output_queue, best_obj, lock),
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

            while not output_queue.empty():
                self.processed_cluster_sizes.append(output_queue.get())

            raise KeyboardInterrupt

        logger.info(
            f"Branch&Bound for cluster sizes took {time() - start:.3f} seconds with "
            f"a final objective of {best_obj.value}."
        )

        while not output_queue.empty():
            self.processed_cluster_sizes.append(output_queue.get())

        best_tmp_obj, _, best_sizes = min(
            self.processed_cluster_sizes,
            key=lambda x: x[0] if not isinstance(x[0], str) else np.inf,
        )
        logger.info(f"best_tmp_obj: {best_tmp_obj}, best_sizes: {best_sizes}")

        assert not np.isinf(
            best_tmp_obj
        ), "No feasible solution was found during Branch&Bound."

        if not np.isclose(best_tmp_obj, best_obj.value):
            logger.error(
                f"Best objective found during Branch&Bound {best_tmp_obj} "
                "does not match objective found in run without "
                f"tolerance {best_obj.value}."
            )
        return np.array(best_sizes), best_obj.value

    def sort_labels(self, kmeanspp_labels: np.ndarray) -> Tuple[List[int], List[int]]:
        _, cluster_sizes = np.unique(kmeanspp_labels, return_counts=True)
        sorted_sizes = sorted(cluster_sizes, reverse=True)
        logger.info(f"KMeans++ cluster sizes: {sorted_sizes}")

        sorted_map = {v: i for i, v in enumerate(np.argsort(-cluster_sizes))}

        initial_labels = [sorted_map[ll] for ll in kmeanspp_labels]

        return initial_labels, sorted_sizes

    def compute_initial_cost_bound(self) -> Tuple[float, np.ndarray]:
        best_inertia = np.inf
        best_labels = None

        for i in range(self.kmeans_iterations):
            kmeans = KMeans(
                n_clusters=self.k, n_init="auto", init="k-means++", random_state=i
            )
            kmeans.fit(self.X)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = kmeans.labels_

        if best_labels is None:
            raise ValueError("KMeans could not find a solution.")

        return best_inertia, best_labels

    def optimize(
        self,
        kmeanspp_cost: Optional[float] = None,
        kmeanspp_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:

        if kmeanspp_cost is None and kmeanspp_labels is None:
            kmeanspp_cost, kmeanspp_labels = self.compute_initial_cost_bound()
        elif kmeanspp_labels is None and kmeanspp_cost is not None:
            raise ValueError(
                "If kmeanspp_cost is provided, kmeanspp_labels must be provided as well."
            )
        elif kmeanspp_labels is not None and kmeanspp_cost is None:
            kmeanspp_cost = kmeans_cost(kmeanspp_labels, points=self.X, k=self.k)

        assert (
            kmeanspp_labels is not None
        ), "KMeans++ labels must be either provided or computed before continuing."

        logger.info("Chosen initial KMeans++ solution with cost: %f", kmeanspp_cost)

        try:
            initial_labels, kmeanspp_sizes = self.sort_labels(kmeanspp_labels)
            self.kmeanspp_cluster_cost = kmeanspp_cost

            self.compute_cluster_size_objectives()

            # Construct lower bounds for clustering sizes using the dynamic program
            if self.dp_bounds.sum() == 0:
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
                "dp_bounds": self.dp_bounds.tolist(),
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
        start = time()
        self.model = self.run_fixed_cluster_sizes_ilp(
            cluster_sizes=best_sizes,
            cost=None,
            remove_tolerance=True,
        )

        logger.info(f"Final ILP took {time() - start:.3f} seconds.")

        if not np.isclose(self.model.ObjVal, best_obj, atol=1e-04):
            logger.error(
                f"Objective value of final model {self.model.ObjVal} "
                f"does not match best objective value {best_obj}."
            )

        logger.info(
            f"The best found objective was {self.model.ObjVal} with size {best_sizes} "
            f"compared to initial bound {self.kmeanspp_cluster_cost}."
        )

        labels = self.get_labels()

        return {
            "initial_labels": initial_labels,
            "labels": labels,
            "model": self.model,
            "objective": self.model.ObjVal,
            "best_cluster_sizes": best_sizes,
            "cluster_size_objectives": self.cluster_size_objectives,
            "processed_cluster_sizes": self.processed_cluster_sizes,
        }

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
