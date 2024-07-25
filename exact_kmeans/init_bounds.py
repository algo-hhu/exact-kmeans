import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from kmeans_ilp.util import get_distance, kmeans_cluster_sizes
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class KMeans_bounded:
    def __init__(
        self,
        n_clusters: int,
        kmeans_iterations: int = 100,
        LB: Optional[List] = None,
        UB: Optional[List] = None,
        version: str = "v1",
    ) -> None:
        self.k = n_clusters
        self.kmeans_iterations = kmeans_iterations
        self.version = version

        self.LB = [0] * self.k if LB is None else LB
        self.UB = [np.inf] * self.k if UB is None else UB

        if len(self.LB) != self.k or len(self.UB) != self.k:
            raise ValueError(
                "Number of lower and upper bounds does not match the number of clusters."
            )

    def extract_index(self, name: str) -> int:
        if name[0:2] == "v'":
            return int(name[2:])
        return int(name[1:])

    def kmeans_cluster_sizes(self, kmeans_labels: np.ndarray) -> Dict:
        labels, sizes = np.unique(kmeans_labels, return_counts=True)
        cluster_sizes = {labels[i]: sizes[i] for i in range(len(labels))}
        return cluster_sizes

    # min-cost-flow algorithm can cause errors for non-integer weights
    # therefore the distances between centers are rounded
    def center_dist_rounded(self, cluster_centers: np.ndarray) -> np.ndarray:
        distances = np.zeros((self.k, self.k))

        for i in range(self.k):
            for j in range(i + 1, self.k):
                distances[i, j] = get_distance(cluster_centers[i], cluster_centers[j])
                distances[j, i] = distances[i, j]

        min_dist = np.min(distances[np.nonzero(distances)])
        min_dist = min_dist if min_dist != 0 else 0.0001

        distances = np.floor(1000 * distances / min_dist)
        return distances

    def sanity_check(self, cluster_labels: np.ndarray) -> None:
        cluster_sizes = kmeans_cluster_sizes(cluster_labels)
        _, success = self.check_bound_feasibility(cluster_sizes)
        if success is False:
            raise ValueError("Some lower or upper bounds are still violated...")

    def kmeans_cost(self, cluster_labels: np.ndarray) -> float:
        dim = self.X.shape[1]
        centroids = np.zeros(shape=(self.k, dim))
        sizes = np.zeros(self.k)

        # computation of centroids
        for i in range(self.n):
            label = cluster_labels[i]
            centroids[label] += self.X[i]
            sizes[label] += 1

        for i in range(self.k):
            centroids[i] /= sizes[i]

        cost = 0
        for i in range(self.n):
            label = cluster_labels[i]
            cost += get_distance(centroids[label], self.X[i])

        return cost

    def establish_bound_feasibility_v1(
        self,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_sizes: Dict,
        cluster_bounds: Dict,
    ) -> None:
        G = nx.DiGraph()
        distances = self.center_dist_rounded(cluster_centers)
        for label_i in cluster_sizes:
            size = cluster_sizes[label_i]
            LB = cluster_bounds[label_i][0]
            UB = cluster_bounds[label_i][1]

            # cluster_stats = bound_assign[i]
            # violated = cluster_stats[0]
            # LB = cluster_stats[1][0]
            # UB = cluster_stats[1][1]

            if size < LB:
                # add edges to sink
                G.add_edge(f"v{label_i}", "t", capacity=LB - size, weight=0)
                G.add_edge(f"v'{label_i}", "t", capacity=UB - LB, weight=0)
            elif size > UB:
                for label_j in cluster_sizes:
                    dist_ij = distances[label_i][label_j]
                    size_j = cluster_sizes[label_j]
                    LB_j = cluster_bounds[label_j][0]
                    UB_j = cluster_bounds[label_j][1]
                    if size_j < LB_j:
                        G.add_edge(
                            f"v{label_i}",
                            f"v{label_j}",
                            capacity=np.inf,
                            weight=dist_ij,
                        )
                        G.add_edge(
                            f"v{label_i}",
                            f"v'{label_j}",
                            capacity=np.inf,
                            weight=dist_ij,
                        )
                        G.add_edge(
                            f"v'{label_i}",
                            f"v{label_j}",
                            capacity=np.inf,
                            weight=dist_ij,
                        )
                    if LB_j <= size_j and size_j <= UB_j:
                        # add edges to vertices with satisfied bounds
                        G.add_edge(
                            f"v{label_i}",
                            f"v{label_j}",
                            capacity=np.inf,
                            weight=dist_ij,
                        )
                # add edges to source
                G.add_edge("s", f"v{label_i}", capacity=size - UB, weight=0)
                G.add_edge("s", f"v'{label_i}", capacity=UB - LB, weight=0)

            else:
                # add edges to source and sink
                G.add_edge(f"v{label_i}", "t", capacity=UB - size, weight=0)
                G.add_edge("s", f"v'{label_i}", capacity=size - LB, weight=0)
                for label_j in cluster_sizes:
                    size_j = cluster_sizes[label_j]
                    LB_j = cluster_bounds[label_j][0]
                    if size_j < LB_j:
                        # add edges to vertices with violated lower bound
                        dist_ij = distances[label_i][label_j]
                        G.add_edge(
                            f"v'{label_i}",
                            f"v{label_j}",
                            capacity=np.inf,
                            weight=dist_ij,
                        )

        # for e in G.edges:
        #    print(f"Added edge {e} with parameters {G.get_edge_data(u=e[0], v=e[1])}")

        min_cost_flow = nx.max_flow_min_cost(G, "s", "t")

        # print(min_cost_flow)
        # transform flow to reassignemnt of points

        clusters_by_labels: Dict[int, list] = {i: [] for i in range(self.k)}
        for j in range(self.n):
            label = cluster_labels[j]
            clusters_by_labels[label] += [j]

        for e in G.edges:
            x = e[0]
            y = e[1]
            if x == "s" or y == "t":
                continue
            i = self.extract_index(x)
            j = self.extract_index(y)
            for _ in range(int(min_cost_flow[x][y])):
                point = clusters_by_labels[i].pop()
                clusters_by_labels[j].append(point)
                cluster_labels[point] = j

    # check if solution satisfies lower and upper bounds by computing a min cost flow
    def check_bound_feasibility(self, cluster_sizes: Dict) -> Tuple[Dict, bool]:
        G = nx.DiGraph()
        # add edges between clusters and bounds
        # the cost depends on the violation of the bounds by the cluster size
        for i in range(self.k):
            G.add_node(f"w{i}", demand=1)

        for label in cluster_sizes:
            G.add_node(f"v{label}", demand=-1)
            size = cluster_sizes[label]
            for j in range(self.k):
                # violation is zero if bounds are satisfied
                violation = max(0, self.LB[j] - size, size - self.UB[j])
                G.add_edge(f"v{label}", f"w{j}", capacity=1, weight=violation)

        # Compute the min cost flow
        flow = nx.min_cost_flow(G)
        flow_value = nx.cost_of_flow(G, flow)

        logger.info(
            "In check_bound_feasibility: "
            f"Found flow with flow value {flow_value}: {flow}"
        )

        # transform flow into cluster bounds assignment
        cluster_bounds = {label: Tuple[int, int] for label in cluster_sizes}
        for label in cluster_sizes:
            check = False
            size = cluster_sizes[label]
            for j in range(self.k):
                if flow[f"v{label}"][f"w{j}"] == 1:
                    # print(f"cluster {i} of size {size} is"
                    #       f"assigend LB {LB[j]} and UB {UB[j]}\n")
                    check = True
                    if size < self.LB[j]:
                        cluster_bounds[label] = [self.LB[j], self.UB[j]]
                    elif size > self.UB[j]:
                        cluster_bounds[label] = [self.LB[j], self.UB[j]]
                    else:
                        cluster_bounds[label] = [self.LB[j], self.UB[j]]
            if check is False:
                raise ValueError("Computed invalid flow ...")
        if flow_value == 0:
            return cluster_bounds, True

        return cluster_bounds, False

    def establish_bounds(
        self, kmeans_labels: np.ndarray, kmeans_centers: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        # compute assignemnt of cluster sizes to bounds via a min cost flow
        cluster_sizes = kmeans_cluster_sizes(kmeans_labels)

        if len(cluster_sizes) < self.k:
            logger.info("KMeans++ solution with less than k clusters, skipping...")

        cluster_bounds, feasible = self.check_bound_feasibility(cluster_sizes)
        if feasible:
            cost_bounded = self.kmeans_cost(kmeans_labels)
            return (cost_bounded, kmeans_labels)

        # if bounds could not be satisfied, establish bounds via min cost flow
        if self.version == "v1":
            self.establish_bound_feasibility_v1(
                kmeans_labels, kmeans_centers, cluster_sizes, cluster_bounds
            )

        self.sanity_check(kmeans_labels)

        cost_bounded = self.kmeans_cost(kmeans_labels)

        return (cost_bounded, kmeans_labels)

    def fit(self, X: np.ndarray) -> "KMeans_bounded":
        self.X = X
        self.n = len(X)
        self.best_inertia = np.inf
        self.best_labels = None

        # if no bounds provided run vanilla kmeans++
        # TODO adapt
        if self.LB == [0] * self.k and self.UB == [np.inf] * self.k:
            for i in range(self.kmeans_iterations):
                kmeans = KMeans(
                    n_clusters=self.k, n_init="auto", init="k-means++", random_state=i
                )
                kmeans.fit(self.X)
                if kmeans.inertia_ < self.best_inertia:
                    self.best_inertia = kmeans.inertia_
                    self.best_labels = kmeans.labels_

        # if bounds provided run vanilla kmeans++ and modify solution until bounds are satisfied
        else:
            for i in range(self.kmeans_iterations):
                kmeans = KMeans(
                    n_clusters=self.k, n_init="auto", init="k-means++", random_state=i
                )
                kmeans.fit(self.X)
                # if the solution is already worse or we have less than k clusters we stop here
                if kmeans.inertia_ >= self.best_inertia:
                    continue

                cost_bounded, labels_bounded = self.establish_bounds(
                    kmeans.labels_, kmeans.cluster_centers_
                )

                if cost_bounded < self.best_inertia:
                    self.best_inertia = cost_bounded
                    self.best_labels = labels_bounded

        return self
