import argparse
import json
import logging
import sys
from pathlib import Path
from time import time

import pandas as pd

from exact_kmeans.ilp import ExactKMeans
from exact_kmeans.util import JsonEncoder, get_git_hash

logger = logging.getLogger(__name__)


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
    parser.add_argument("--config-file", type=Path, default="config/default.yaml")
    parser.add_argument("--kmeans-iterations", type=int, default=100)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--load-existing-run-path", type=Path, default=None)
    parser.add_argument("--cache-current-run-path", type=Path, default=None)

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

    ilp = ExactKMeans(
        X=X,
        k=args.k,
        config_file=args.config_file,
        load_existing_run_path=args.load_existing_run_path,
        cache_current_run_path=args.cache_current_run_path,
        kmeans_iterations=args.kmeans_iterations,
    )

    start = time()
    res = ilp.optimize()
    ilp_time = time() - start

    if args.results_path is not None:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        model = res["model"]
        res_eval = {
            "version": ilp.ilp_version,
            "time": ilp_time,
            "initial_labels": res["initial_labels"],
            "labels": res["labels"],
            "objective": res["objective"],
            "cluster_size_objectives": res["cluster_size_objectives"],
            "best_cluster_sizes": res["best_cluster_sizes"],
            "processed_cluster_sizes": res["processed_cluster_sizes"],
            "changed_model_params": ilp.changed_model_params,
            "changed_bound_params": ilp.changed_bound_model_params,
            "optimal": model.Status == 2,
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
            res_eval[var_name] = getattr(model, var)

        with open(args.results_path, "w") as f:
            json.dump(res_eval, f, indent=4, cls=JsonEncoder)

        ilp.print_model(
            args.results_path.parent, args.results_path.name.replace(".json", "")
        )
