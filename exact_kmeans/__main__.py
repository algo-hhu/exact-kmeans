import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from exact_kmeans.ilp import ExactKMeans
from exact_kmeans.util import JsonEncoder, get_git_hash

logger = logging.getLogger(__name__)


def read_bounds(
    k: int, n: int, bounds_path: Optional[Path]
) -> Tuple[Optional[List], Optional[List]]:

    if bounds_path is not None:
        LB = []
        UB = []
        with open(bounds_path, newline="") as bounds:
            bounds_reader = csv.reader(bounds, delimiter=",")
            for row in bounds_reader:
                if row[0] == "LB":
                    LB = [int(entry) for entry in row[1:]]
                    if len(LB) < k:
                        logger.info(
                            f"Less that {k} lower bounds provided. Fill up with zeros."
                        )
                        LB.extend([0] * (k - len(LB)))
                if row[0] == "UB":
                    UB = [int(entry) for entry in row[1:]]
                    if len(UB) < k:
                        logger.info(
                            f"Less that {k} upper bounds provided. Fill up with infinity."
                        )
                        UB.extend([np.inf] * (k - len(UB)))

            logger.info(
                f"Computed solution will satisfy lower bounds: {LB} "
                f"and upper bounds: {UB}"
            )
        # check if bounds can be satisfied
        for i in range(k):
            if LB[i] > UB[i]:
                raise ValueError(
                    f"For {i}th pair of bounds:"
                    "lower bound has to be less or equal upper bound."
                )
        if sum(LB) > n:
            raise ValueError("Sum of lower bounds is larger than number of points.")
        if sum(UB) < n:
            raise ValueError("Sum of upper bounds is smaller than number of points.")

    else:
        LB = None
        UB = None

    return (LB, UB)


def set_up_logger(log_file: Path, mode: str = "w+") -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s: %(levelname)s/%(filename)s:%(lineno)d] %(message)s"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--config-file", type=Path, default="exact_kmeans/config/default.yaml"
    )
    parser.add_argument("--bounds-path", type=Path, default=None)
    parser.add_argument("--kmeans-iterations", type=int, default=100)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--load-existing-run-path", type=Path, default=None)
    parser.add_argument("--cache-current-run-path", type=Path, default=None)
    parser.add_argument("--bounds-version", type=str, default=None)
    parser.add_argument("--outlier", type=int, default=None)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    set_up_logger(
        args.results_path.parent / args.results_path.name.replace(".json", ".log"),
    )

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    data = pd.read_csv(args.data_path)
    X = data.values
    logger.info(f"The data has the following shape: {X.shape}")

    LB, UB = read_bounds(args.k, X.shape[0], args.bounds_path)

    ilp = ExactKMeans(
        n_clusters=args.k,
        config_file=args.config_file,
        LB=LB,
        UB=UB,
        # outlier=args.outlier,
        load_existing_run_path=args.load_existing_run_path,
        cache_current_run_path=args.cache_current_run_path,
        kmeans_iterations=args.kmeans_iterations,
        # bounds_version=args.bounds_version,
    )

    start = time()
    ilp.fit(X)
    ilp_time = time() - start

    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        res_eval = {
            "version": ilp.ilp_version,
            "time": ilp_time,
            "initial_labels": ilp.initial_labels,
            "labels": ilp.labels_,
            "optimal_objective": ilp.inertia_,
            "cluster_size_objectives": ilp.cluster_size_objectives,
            "best_cluster_sizes": ilp.best_cluster_sizes,
            "processed_cluster_sizes": ilp.processed_cluster_sizes,
            "changed_model_params": ilp.changed_model_params,
            "changed_bound_params": ilp.changed_bound_model_params,
            "optimal": ilp.model.Status == 2 if ilp.model is not None else False,
            "git_hash": get_git_hash(),
        }
        for var_name, var in [
            ("gap", "MIPGap"),
            ("num_constraints", "NumConstrs"),
            ("num_variables", "NumVars"),
            ("upper_bound", "ObjVal"),
            ("lower_bound", "ObjBound"),
            ("solver_time", "Runtime"),
        ]:
            res_eval[var_name] = getattr(ilp.model, var)

        with open(args.results_path, "w") as f:
            json.dump(res_eval, f, indent=4, cls=JsonEncoder)

        ilp.print_model(
            args.results_path.parent, args.results_path.name.replace(".json", "")
        )
