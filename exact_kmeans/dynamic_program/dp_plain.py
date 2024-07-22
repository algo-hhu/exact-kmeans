import math
from typing import Dict

import numpy as np


def DP_init(n_points: int, k: int, lb: Dict[int, float]) -> np.ndarray:
    # make array one dimension larger to avoid substracting 1 every time
    r = np.zeros(shape=(n_points + 1, k + 1), dtype=float)

    r[0][0] = 0
    r[1][0] = math.inf
    r[0][1] = math.inf

    for covered in range(1, n_points + 1):
        if covered in lb:
            r[covered][1] = lb[covered]
        else:
            r[covered][1] = math.inf
        r[covered][0] = math.inf

    for clusters in range(2, k + 1):
        r[0][clusters] = math.inf
        r[1][clusters] = math.inf

    # print(r)
    return r


def DP_recursion(n_points: int, k: int, lb: Dict[int, float], r: np.ndarray) -> None:
    for covered in range(2, n_points + 1):
        for clusters in range(2, k + 1):
            max_size = min(covered - clusters + 1, n_points - k + 1)
            # print(f"Covered: {covered}, Clusters: {clusters}, max_size= {max_size}")
            min_curr = math.inf

            for cluster_size in range(1, max_size + 1):
                if cluster_size in lb:
                    min_new = r[covered - cluster_size][clusters - 1] + lb[cluster_size]
                else:
                    min_new = math.inf

                if min_new < min_curr:
                    min_curr = min_new

            r[covered][clusters] = min_curr


def DP_recursion_print_part(
    n_points: int, k: int, lb: Dict[int, float], r: np.ndarray
) -> None:
    partition = {(i, 1): [i] for i in range(0, n_points + 1)}

    for covered in range(2, n_points + 1):
        for clusters in range(2, k + 1):
            max_size = min(covered - clusters + 1, n_points - k + 1)
            # print(f"Covered: {covered}, Clusters: {clusters}, max_size= {max_size}")
            min_curr = math.inf
            min_size = 0

            for cluster_size in range(1, max_size + 1):
                if cluster_size in lb:
                    min_new = r[covered - cluster_size][clusters - 1] + lb[cluster_size]
                else:
                    min_new = math.inf

                if min_new < min_curr:
                    min_curr = min_new
                    min_size = cluster_size

            r[covered][clusters] = min_curr
            new_part = partition[(covered - min_size, clusters - 1)][:]
            new_part.append(min_size)
            partition.update({(covered, clusters): new_part})

    for i in range(1, n_points + 1):
        print(f"Number of points {i}:")
        for j in range(1, min(k + 1, i + 1)):
            curr = partition[(i, j)]
            print(f"{curr} ", end="")
        print("\n")


def compute_bounds(
    n_points: int, k: int, lb: Dict[int, float], print_part: bool = False
) -> np.ndarray:
    if n_points == 0 or k == 0:
        raise ValueError(
            f"Number of points (n={n_points}) "
            f"and number of clusters (k={k}) must be greater than 0"
        )
    r = DP_init(n_points, k, lb)
    if print_part:
        DP_recursion_print_part(n_points, k, lb, r)
    else:
        DP_recursion(n_points, k, lb, r)

    return r
