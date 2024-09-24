import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from exact_kmeans.util import compute_center_distances, get_distance

logger = logging.getLogger(__name__)


class KMeans_vanilla:
    def __init__(self, n_clusters: int, kmeans_iterations: int = 100) -> None:
        self.k = n_clusters
        self.kmeans_iterations = kmeans_iterations

    def fit(self, X: np.ndarray) -> "KMeans_vanilla":
        self.X = X
        self.n = len(X)
        self.best_inertia = np.inf
        self.best_labels = None

        for i in range(self.kmeans_iterations):
            kmeans = KMeans(
                n_clusters=self.k, n_init="auto", init="k-means++", random_state=i
            )
            kmeans.fit(self.X)
            if kmeans.inertia_ < self.best_inertia:
                self.best_inertia = kmeans.inertia_
                self.best_labels = kmeans.labels_
        return self


class KMeans_outlier:
    def __init__(
        self, n_clusters: int, kmeans_iterations: int = 100, outlier: int = 0
    ) -> None:
        self.k = n_clusters
        self.kmeans_iterations = kmeans_iterations
        self.outlier = outlier
        if self.outlier < 0:
            raise ValueError("Number of outliers must be positive.")
        if self.outlier == 0:
            logger.info(
                "Called initialization of KMeans with outliers with 0 outliers."
                "Compute vanilla KMeans solution."
            )

        self.out_label = n_clusters

    def compute_assignment(self, cluster_centers: np.ndarray) -> np.ndarray:
        labels = np.zeros(shape=self.n, dtype=int)
        new_k = len(cluster_centers)

        for i in range(self.n):
            min = np.inf
            for j in range(new_k):
                dist = get_distance(cluster_centers[j], self.X[i])
                if dist < min:
                    min = dist
                    labels[i] = j

        return labels

    def compute_centroids(self, cluster_labels: np.ndarray) -> Tuple[np.ndarray, float]:
        dim = self.X.shape[1]
        centroids = np.zeros(shape=(self.k, dim))

        for i in range(self.k):
            id = np.where(cluster_labels == i)[0]
            if np.any(id):
                centroids[i] = self.X[id].mean(axis=0)

        cost = 0
        for i in range(self.n):
            label = cluster_labels[i]
            if label != self.out_label:
                cost += get_distance(centroids[label], self.X[i])

        return (centroids, cost)

    def remove_outlier(
        self, kmeans_labels: np.ndarray, kmeans_centers: np.ndarray
    ) -> Tuple[float, np.ndarray]:

        min_cost = np.inf
        for i in range(3):
            if i > 0:
                kmeans_labels = self.compute_assignment(kmeans_centers)
            dist = compute_center_distances(self.X, kmeans_labels, kmeans_centers)
            dist.sort(key=lambda item: item[1], reverse=True)
            # remove the points that are farthest away from the centers
            for i in range(self.outlier):
                pt = dist[i][0]
                kmeans_labels[pt] = self.out_label

            kmeans_centers, kmeans_cost = self.compute_centroids(kmeans_labels)
            if kmeans_cost < min_cost:
                min_cost = kmeans_cost

        return kmeans_cost, kmeans_labels

    def fit(self, X: np.ndarray) -> "KMeans_outlier":
        self.X = X
        self.n = len(X)
        self.best_inertia = np.inf
        self.best_labels = None

        if self.outlier > self.n:
            raise ValueError(
                f"Number of outliers {self.outlier} exceeds number of points {self.n}."
            )

        for i in range(self.kmeans_iterations):
            kmeans = KMeans(
                n_clusters=self.k, n_init="auto", init="k-means++", random_state=i
            )
            kmeans.fit(self.X)
            if self.outlier > 0:
                cost_outlier, labels_outlier = self.remove_outlier(
                    kmeans_labels=kmeans.labels_, kmeans_centers=kmeans.cluster_centers_
                )
            else:
                cost_outlier = kmeans.inertia_
                labels_outlier = kmeans.labels_

            if cost_outlier < self.best_inertia:
                self.best_inertia = cost_outlier
                self.best_labels = labels_outlier

        return self


class KMeans_bounded:
    def __init__(
        self,
        n_clusters: int,
        kmeans_iterations: int = 100,
        LB: Optional[List] = None,
        UB: Optional[List] = None,
        outlier: int = 0,
    ) -> None:
        self.k = n_clusters
        self.kmeans_iterations = kmeans_iterations

        self.LB = [0] * self.k if LB is None else LB
        self.UB = [np.inf] * self.k if UB is None else UB
        self.outlier = outlier
        self.out_label = n_clusters
        self.uniform = False
        if len(set(self.UB)) == 1 and len(set(self.LB)) == 1:
            self.uniform = True

        if self.outlier < 0:
            raise ValueError("Number of outliers must be positive.")

        if len(self.LB) != self.k or len(self.UB) != self.k:
            raise ValueError(
                "Number of lower and upper bounds does not match the number of clusters."
            )

    def extract_index(self, name: str) -> int:
        if name == "out":
            return self.out_label
        if name[0:2] == "v'" or name[0:2] == "w'":
            return int(name[2:])
        return int(name[1:])

    def kmeans_cluster_sizes(self, kmeans_labels: np.ndarray) -> Dict:
        labels, sizes = np.unique(kmeans_labels, return_counts=True)
        cluster_sizes = {labels[i]: sizes[i] for i in range(len(labels))}
        return cluster_sizes

    # min-cost-flow algorithm can cause errors for non-integer weights
    # therefore the distances between centers are rounded
    def dist_rounded(self, S: np.ndarray, T: np.ndarray) -> np.ndarray:
        s = S.shape[0]
        t = T.shape[0]
        distances = np.zeros((s, t))

        for i in range(s):
            for j in range(t):
                distances[i, j] = get_distance(S[i], T[j])

        min_dist = np.min(distances[np.nonzero(distances)])
        min_dist = min_dist if min_dist != 0 else 0.0001

        distances = np.floor(1000 * distances / min_dist)
        return distances

    def sanity_check(self, cluster_labels: np.ndarray) -> None:
        cluster_sizes = self.kmeans_cluster_sizes(cluster_labels)
        _, success = self.check_bound_feasibility(cluster_sizes)
        if success is False:
            raise ValueError("Some lower or upper bounds are still violated...")

    def compute_assignment(
        self, cluster_centers: np.ndarray
    ) -> Tuple[np.array, np.ndarray]:
        labels = np.zeros(shape=self.n, dtype=int)
        new_k = len(cluster_centers)
        cluster_sizes = {j: 0 for j in range(new_k)}

        for i in range(self.n):
            min = np.inf
            for j in range(new_k):
                dist = get_distance(cluster_centers[j], self.X[i])
                if dist < min:
                    min = dist
                    labels[i] = j

        for i in range(self.n):
            cluster_sizes[labels[i]] += 1

        return (labels, cluster_sizes)

    def compute_centroids(self, cluster_labels: np.ndarray) -> np.ndarray:
        dim = self.X.shape[1]
        centroids = np.zeros(shape=(self.k, dim))
        for i in range(self.k):
            id = np.where(cluster_labels == i)[0]
            if np.any(id):
                centroids[i] = self.X[id].mean(axis=0)

        return centroids

    def kmeans_cost(self, cluster_labels: np.ndarray) -> float:
        centroids = self.compute_centroids(cluster_labels)
        cost = 0

        for i in range(self.n):
            label = cluster_labels[i]
            if label != self.out_label:
                cost += get_distance(centroids[label], self.X[i])
        return cost

    def establish_bound_feasibility(
        self,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_sizes: Dict,
        cluster_bounds: Dict,
    ) -> None:
        G = nx.DiGraph()
        distances = self.dist_rounded(self.X, cluster_centers)

        for i in range(self.n):
            G.add_edge("s", f"v{i}", capacity=1, weight=0)
            G.add_edge(f"v{i}", "out", capacity=1, weight=0)
            for label in cluster_sizes:
                dist = distances[i][label]
                G.add_edge(f"v{i}", f"w{label}", capacity=1, weight=dist)
                G.add_edge(f"v{i}", f"w'{label}", capacity=1, weight=dist)

        for label in cluster_sizes:
            LB = cluster_bounds[label][0]
            UB = cluster_bounds[label][1]

            G.add_edge(f"w{label}", "t_l", capacity=LB, weight=0)
            G.add_edge(f"w'{label}", "t_uml", capacity=UB - LB, weight=0)

        LB_sum = sum(self.LB)
        max_outlier = min(self.outlier, self.n - LB_sum)
        G.add_edge("out", "t", capacity=max_outlier, weight=0)
        G.add_edge("t_l", "t", capacity=LB_sum, weilght=0)
        G.add_edge("t_uml", "t", capacity=self.n - LB_sum - max_outlier, weight=0)

        min_cost_flow = nx.max_flow_min_cost(G, "s", "t")

        # transform flow to reassignemnt of points

        check = 0
        for e in G.edges:
            x = e[0]
            y = e[1]
            if x == "s" or y in ["t", "t_l", "t_uml"]:
                continue

            i = self.extract_index(x)
            j = self.extract_index(y)
            if min_cost_flow[x][y] == 1:
                check += 1
                cluster_labels[i] = j

        if check != self.n:
            raise ValueError("Computed invalid flow.")

    # check if solution satisfies lower and upper bounds by computing a min cost flow
    def check_bound_feasibility(self, cluster_sizes: Dict) -> Tuple[Dict, bool]:
        cluster_bounds = {label: Tuple[int, int] for label in cluster_sizes}
        if self.uniform:
            excess = 0
            LB = self.LB[0]
            UB = self.UB[0]
            # all cluster sizes are assignet the same bounds
            LB_violated = False
            for label in cluster_sizes:
                cluster_bounds[label] = (LB, UB)
                if LB > cluster_sizes[label]:
                    LB_violated = True
                excess += max(0, cluster_sizes[label] - UB)
            if excess > self.outlier or LB_violated:
                return cluster_bounds, False
            else:
                return cluster_bounds, True

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

        if len(cluster_sizes) != self.k:
            diff = self.k - len(cluster_sizes)
            G.add_node("z", demand=-diff)
            for j in range(self.k):
                G.add_edge("z", f"w{j}", capacity=1, weight=0)

        # Compute the min cost flow
        flow = nx.min_cost_flow(G)

        # transform flow into cluster bounds assignment
        excess = 0
        LB_violated = False
        for label in cluster_sizes:
            check = False
            size = cluster_sizes[label]
            for j in range(self.k):
                if flow[f"v{label}"][f"w{j}"] == 1:
                    check = True
                    if size < self.LB[j]:
                        LB_violated = True
                    size_excess = size - self.UB[j]
                    if size_excess > 0:
                        excess += size_excess
                    cluster_bounds[label] = [self.LB[j], self.UB[j]]
            if check is False:
                raise ValueError("Computed invalid flow.")
        if LB_violated is False and excess <= self.outlier:
            return cluster_bounds, True

        return cluster_bounds, False

    def remove_outlier(
        self,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_sizes: Dict,
        cluster_bounds: Dict,
    ) -> None:
        dist = compute_center_distances(self.X, cluster_labels, cluster_centers)

        dist.sort(key=lambda item: item[1], reverse=True)

        removed = 0
        # first remove points whose cluster violates the upper bound
        for i in range(self.n):
            if removed >= self.outlier:
                break
            point = dist[i][0]
            label = cluster_labels[point]
            if cluster_sizes[label] > cluster_bounds[label][1]:
                cluster_labels[point] = self.out_label
                cluster_sizes[label] -= 1
                removed += 1

        # now remove points if its cluster does not violate the lower bound after removal
        for i in range(self.n):
            if removed >= self.outlier:
                break
            point = dist[i][0]
            label = cluster_labels[point]
            if (
                label != self.out_label
                and cluster_sizes[label] > cluster_bounds[label][0]
            ):
                cluster_labels[point] = self.out_label
                cluster_sizes[label] -= 1
                removed += 1

    def establish_bounds(
        self, kmeans_labels: np.ndarray, kmeans_centers: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        # compute assignemnt of cluster sizes to bounds via a min cost flow
        for _ in range(10):
            kmeans_centers = self.compute_centroids(kmeans_labels)
            kmeans_labels, cluster_sizes = self.compute_assignment(kmeans_centers)
            """
            if len(cluster_sizes) < self.k:
                if len(kmeans_centers) < self.k:
                    logger.info("KMeans solution with less than k clusters, skipping...")
                    cost_bounded = self.kmeans_cost(kmeans_labels)
                    return (cost_bounded, kmeans_labels)
                diff = self.k - len(cluster_sizes)
                for _ in range(diff):
                    cluster_sizes.append(0)
            """
            cluster_bounds, feasible = self.check_bound_feasibility(cluster_sizes)
            if feasible:
                if self.outlier > 0:
                    self.remove_outlier(
                        kmeans_labels, kmeans_centers, cluster_sizes, cluster_bounds
                    )
                cost_bounded = self.kmeans_cost(kmeans_labels)
                return (cost_bounded, kmeans_labels)

            # if bounds could not be satisfied, establish bounds via min cost flow
            self.establish_bound_feasibility(
                kmeans_labels, kmeans_centers, cluster_sizes, cluster_bounds
            )

        # self.sanity_check(kmeans_labels)
        cost_bounded = self.kmeans_cost(kmeans_labels)

        return (cost_bounded, kmeans_labels)

    # checks if cluster sizes satisfy bound constraints
    def fit_cluster_sizes(self, sizes: List) -> bool:
        cluster_sizes = {i: sizes[i] for i in sizes}
        _, success = self.check_bound_feasibility(cluster_sizes)
        return success

    def fit(self, X: np.ndarray) -> "KMeans_bounded":
        self.X = X
        self.n = len(X)
        self.best_inertia = np.inf
        self.best_labels = None

        # if no bounds provided run vanilla kmeans++
        if self.outlier > self.n:
            raise ValueError(
                f"Number of outliers {self.outlier} exceeds number of points {self.n}."
            )

        if self.LB == [0] * self.k and self.UB == [np.inf] * self.k:
            kmeans_out = KMeans_outlier(
                n_clusters=self.k,
                kmeans_iterations=self.kmeans_iterations,
                outlier=self.outlier,
            )
            kmeans_out.fit(self.X)
            self.best_inertia = kmeans_out.best_inertia
            self.best_labels = kmeans_out.best_labels

        # if bounds provided run vanilla kmeans++ and modify solution until bounds are satisfied
        else:
            for i in range(self.kmeans_iterations):
                kmeans = KMeans(
                    n_clusters=self.k,
                    n_init="auto",
                    init="k-means++",
                    random_state=i,
                )
                kmeans.fit(self.X)

                if self.outlier == 0 and kmeans.inertia_ >= self.best_inertia:
                    continue
                cost_bounded, labels_bounded = self.establish_bounds(
                    kmeans.labels_, kmeans.cluster_centers_
                )
                if cost_bounded < self.best_inertia:
                    self.best_inertia = cost_bounded
                    self.best_labels = labels_bounded

        return self
