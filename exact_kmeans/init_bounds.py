import logging
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from kmeans_ilp.util import get_distance, kmeans_cluster_sizes, kmeans_cost
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# check if solution satisfies lower and upper bounds by computing a min cost flow
def check_bound_feasibility(
    cluster_sizes: np.ndarray, LB: List, UB: List, k: int
) -> Tuple[Dict, bool]:
    G = nx.DiGraph()
    # add edges between clusters and bounds
    # the cost depends on the violation of the bounds by the cluster size
    for i in range(k):
        G.add_node(f"v{i}", demand=-1)
        G.add_node(f"w{i}", demand=1)
        size = cluster_sizes[i]
        for j in range(k):
            # violation is zero if bounds are satisfied
            violation = max(0, LB[j] - size, size - UB[j])
            G.add_edge(f"v{i}", f"w{j}", capacity=1, weight=violation)

    # Compute the min cost flow
    flow = nx.min_cost_flow(G)
    flow_value = nx.cost_of_flow(G, flow)

    logger.info(
        "In check_bound_feasibility: "
        f"Found flow with flow value {flow_value}: {flow}"
    )

    # transform flow into cluster bounds assignment
    bound_assign = {i: Tuple[str, list] for i in range(k)}
    for i in range(k):
        check = False
        size = cluster_sizes[i]
        for j in range(k):
            if flow[f"v{i}"][f"w{j}"] == 1:
                # print(f"cluster {i} of size {size} is assigend LB {LB[j]} and UB {UB[j]}\n")
                check = True
                if size < LB[j]:
                    bound_assign[i] = ("LB", [LB[j], UB[j]])
                elif size > UB[j]:
                    bound_assign[i] = ("UB", [LB[j], UB[j]])
                else:
                    bound_assign[i] = ("NO", [LB[j], UB[j]])
        if check is False:
            logger.info("Error:invalid flow....")
    if flow_value == 0:
        return bound_assign, True

    return bound_assign, False


def extract_index(name: str) -> int:
    if name[0:2] == "v'":
        return int(name[2:])
    return int(name[1:])


# min-cost-flow algorithm can cause errors for non-integer weights
# therefore the distances between centers are rounded
def center_dist_rounded(cluster_centers: np.ndarray) -> np.ndarray:
    k = len(cluster_centers)
    distances = np.zeros((k, k))

    for i in range(k):
        for j in range(i + 1, k):
            distances[i, j] = get_distance(cluster_centers[i], cluster_centers[j])
            distances[j, i] = distances[i, j]

    min_dist = np.min(distances[np.nonzero(distances)])
    min_dist = min_dist if min_dist != 0 else 0.0001

    distances = np.floor(1000 * distances / min_dist)

    return distances


def establish_bound_feasibility(
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    cluster_sizes: np.ndarray,
    bound_assign: Dict,
    k: int,
) -> None:
    G = nx.DiGraph()
    distances = center_dist_rounded(cluster_centers)
    for i in range(k):
        size = cluster_sizes[i]
        cluster_stats = bound_assign[i]
        violated = cluster_stats[0]
        LB = cluster_stats[1][0]
        UB = cluster_stats[1][1]

        if violated == "LB":
            # add edges to sink
            G.add_edge(f"v{i}", "t", capacity=LB - size, weight=0)
            G.add_edge(f"v'{i}", "t", capacity=UB - LB, weight=0)
        if violated == "UB":
            for j in range(k):
                dist = distances[i][j]
                if bound_assign[j][0] == "LB":
                    G.add_edge(f"v{i}", f"v{j}", capacity=np.inf, weight=dist)
                    G.add_edge(f"v{i}", f"v'{j}", capacity=np.inf, weight=dist)
                    G.add_edge(f"v'{i}", f"v{j}", capacity=np.inf, weight=dist)
                if bound_assign[j][0] == "NO":
                    # add edges to vertices with satisfied bounds
                    G.add_edge(f"v{i}", f"v{j}", capacity=np.inf, weight=dist)
            # add edges to source
            G.add_edge("s", f"v{i}", capacity=size - UB, weight=0)
            G.add_edge("s", f"v'{i}", capacity=UB - LB, weight=0)

        if violated == "NO":
            # add edges to source and sink
            G.add_edge(f"v{i}", "t", capacity=UB - size, weight=0)
            G.add_edge("s", f"v'{i}", capacity=size - LB, weight=0)
            for j in range(k):
                if bound_assign[j][0] == "LB":
                    # add edges to vertices with violated lower bound
                    dist = distances[i][j]
                    G.add_edge(f"v'{i}", f"v{j}", capacity=np.inf, weight=dist)

    # for e in G.edges:
    #    print(f"Added edge {e} with parameters {G.get_edge_data(u=e[0], v=e[1])}")

    min_cost_flow = nx.max_flow_min_cost(G, "s", "t")

    # print(min_cost_flow)
    # transform flow to reassignemnt of points
    n = len(cluster_labels)

    clusters_by_labels: Dict[int, list] = {i: [] for i in range(k)}
    for j in range(n):
        label = cluster_labels[j]
        clusters_by_labels[label] += [j]

    for e in G.edges:
        x = e[0]
        y = e[1]
        if x == "s" or y == "t":
            continue
        i = extract_index(x)
        j = extract_index(y)
        for _ in range(int(min_cost_flow[x][y])):
            point = clusters_by_labels[i].pop()
            clusters_by_labels[j].append(point)
            cluster_labels[point] = j


def sanity_check(cluster_labels: np.ndarray, LB: List, UB: List, k: int) -> None:
    _, cluster_sizes = kmeans_cluster_sizes(cluster_labels)
    _, sucess = check_bound_feasibility(cluster_sizes, LB, UB, k)
    if sucess is False:
        logger.info("Error: Bounds are still infeasible.")


def establish_bounds(
    kmeans_labels: np.ndarray, kmeans_centers: np.ndarray, LB: List, UB: List, k: int
) -> np.ndarray:

    bounded_labels = kmeans_labels

    # compute assignemnt of cluster sizes to bounds via a min cost flow
    cluster_labels, cluster_sizes = kmeans_cluster_sizes(kmeans_labels)

    if len(cluster_labels) < k:
        logger.info("KMeans++ solution with less than k clusters, skipping...")

    bound_assign, feasible = check_bound_feasibility(cluster_sizes, LB, UB, k)
    if feasible:
        return bounded_labels

    # if bounds could not be satisfied, establish bounds via min cost flow
    establish_bound_feasibility(
        bounded_labels, kmeans_centers, cluster_sizes, bound_assign, k
    )
    sanity_check(bounded_labels, LB, UB, k)

    return bounded_labels


def bounded_kMeans_solution(
    k: int, iterations: int, points: np.ndarray, LB: List, UB: List
) -> Tuple[float, np.ndarray]:
    best_inertia = np.inf
    best_labels = None

    # if no bounds provided run vanilla kmeans++
    if LB == [0] * k and UB == [np.inf] * k:
        for i in range(iterations):
            kmeans = KMeans(
                n_clusters=k, n_init="auto", init="k-means++", random_state=i
            )
            kmeans.fit(points)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = kmeans.labels_

    # if bounds provied run vanilla kmeans++ and modify solution until bounds are satisfied
    else:
        for i in range(iterations):
            kmeans = KMeans(
                n_clusters=k, n_init="auto", init="k-means++", random_state=i
            )
            kmeans.fit(points)
            # if the solution is already worse or we have less than k clusters we stop here
            if kmeans.inertia_ >= best_inertia:
                continue

            labels_bounded = establish_bounds(
                kmeans.labels_, kmeans.cluster_centers_, LB, UB, k
            )
            cost_bounded = kmeans_cost(labels_bounded, points, k)

            if cost_bounded < best_inertia:
                best_inertia = cost_bounded
                best_labels = labels_bounded

    return best_inertia, best_labels
