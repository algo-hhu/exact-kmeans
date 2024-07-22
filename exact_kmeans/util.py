import json
import logging
import subprocess
from typing import Any, Generator, List, Optional, Tuple

import gurobipy as gp
import numpy as np

logger = logging.getLogger(__name__)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JsonEncoder, self).default(obj)


def get_git_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_clusters_from_labels(labels: np.ndarray, k: int) -> List[List[int]]:
    clusters: List[List[int]] = [[] for _ in range(k)]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    return clusters


def print_variables(var_list: List[gp.Var], float_list: List[float]) -> Generator:
    current_var: Tuple[Optional[str], Optional[int]] = (None, None)
    for var, val in zip(var_list, float_list):
        if val == 0:
            continue
        inst = var.VarName.split("[")
        var_name = inst[0]
        indices = inst[1][:-1].split(",")
        if current_var != (var_name, len(indices)):
            yield f"\n\nVariable {var_name.upper()} [{len(indices)}]\n"
            current_var = (var_name, len(indices))
        yield print_variable(var_name, indices, val)


def print_variable(var_name: str, indices: List[str], value: float) -> Optional[str]:
    if var_name == "x":
        if len(indices) == 1:
            i = indices[0]
            return f"\npoint i = {i} is assigned to the cluster, value {value}"
        elif len(indices) == 2:
            i, ll = indices
            return f"\npoint i = {i} is assigned to cluster l = {ll}, value {value}"
        else:
            return f"Unknown variable {var_name} with indices {indices}"
    elif var_name == "s":
        ll, m = indices
        return f"\ncluster l = {ll} has size m = {m}, value {value}"
    elif var_name == "y":
        if len(indices) == 2:
            i, j = indices
            return f"\npoints i = {i} and j = {j} both belong to the cluster, value {value}"
        elif len(indices) == 3:
            i, j, m = indices
            return (
                f"\npoints i = {i} and j = {j} with cluster size m = {m}, value {value}"
            )
        else:
            return f"Unknown variable {var_name} with indices {indices}"
    elif var_name == "z":
        if len(indices) == 2:
            i, j = indices
            return f"\npoints i = {i}, j = {j}, value {value}"
        elif len(indices) == 3:
            i, j, m = indices
            return f"\npoints i = {i}, j = {j} with cluster size m = {m}, value {value}"
        elif len(indices) == 4:
            i, j, ll, m = indices
            return (
                f"\npoints i = {i}, j = {j} are in cluster l = {ll} "
                f"with size m = {m}, "
                f"value {value}"
            )
    elif var_name == "w":
        if len(indices) == 3:
            i, ll, m = indices
            return f"\npoint i = {i} is assigned to cluster l = {ll} which has size m = {m}"

    return f"Unknown variable {var_name} with indices {indices}"


def get_distance(x: np.ndarray, y: np.ndarray) -> Any:
    return sum((x - y) ** 2)


def kmeans_cost(cluster_labels: np.ndarray, points: np.ndarray, k: int) -> float:
    dim = points.shape[1]
    centroids = np.zeros(shape=(k, dim))
    sizes = np.zeros(k)

    # computation of centroids
    for i in range(len(points)):
        label = cluster_labels[i]
        centroids[label] += points[i]
        sizes[label] += 1

    for i in range(k):
        centroids[i] /= sizes[i]

    cost = 0
    for i in range(len(points)):
        label = cluster_labels[i]
        cost += get_distance(centroids[label], points[i])

    return cost
