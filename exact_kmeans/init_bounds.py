import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from exact_kmeans.util import get_distance

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


class KMeans_bounded:
    def __init__(
        self,
        n_clusters: int,
        kmeans_iterations: int = 100,
        LB: Optional[List] = None,
        UB: Optional[List] = None,
        version: str = "v2",
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

    def compute_assignment(self, cluster_centers: np.ndarray) -> np.array:
        labels = np.zeros(shape=self.n, dtype=int)

        for i in range(self.n):
            min = np.inf
            for j in range(len(cluster_centers)):
                dist = get_distance(cluster_centers[j], self.X[i])
                if dist < min:
                    min = dist
                    labels[i] = j

        return labels

    def compute_centroids(self, cluster_labels: np.ndarray) -> np.ndarray:
        dim = self.X.shape[1]
        centroids = np.zeros(shape=(self.k, dim))
        sizes = np.zeros(self.k)

        for i in range(self.n):
            label = cluster_labels[i]
            centroids[label] += self.X[i]
            sizes[label] += 1

        for i in range(self.k):
            centroids[i] /= sizes[i]

        return centroids

    def kmeans_cost(self, cluster_labels: np.ndarray) -> float:
        centroids = self.compute_centroids(cluster_labels)
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
        distances = self.dist_rounded(cluster_centers, cluster_centers)

        demand_pos = 0
        demand_neg = 0

        for label_i in cluster_sizes:
            size = cluster_sizes[label_i]
            LB = cluster_bounds[label_i][0]
            UB = cluster_bounds[label_i][1]

            if size < LB:
                # demand
                G.add_node(f"v{label_i}", demand=LB - size)
                demand_pos += LB - size
                G.add_node(f"v'{label_i}", demand=0)

                # add edges to sink
                # G.add_edge(f"v{label_i}", "t", capacity=LB - size, weight=0)
                G.add_edge(f"v'{label_i}", "t", capacity=UB - LB, weight=0)

                # add edge to source
                G.add_edge("s", f"v'{label_i}", capacity=UB - LB, weight=0)
            elif size > UB:
                # demand
                G.add_node(f"v{label_i}", demand=UB - size)
                demand_neg += size - UB
                G.add_node(f"v'{label_i}", demand=0)

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
                # G.add_edge("s", f"v{label_i}", capacity=size - UB, weight=0)
                G.add_edge("s", f"v'{label_i}", capacity=UB - LB, weight=0)
                G.add_edge(f"v'{label_i}", "t", capacity=UB - LB, weight=0)

            else:
                # demand
                G.add_node(f"v{label_i}", demand=0)
                G.add_node(f"v'{label_i}", demand=0)

                # add edges to source and sink
                G.add_edge(f"v{label_i}", "t", capacity=UB - size, weight=0)
                G.add_edge("s", f"v{label_i}", capacity=UB - size, weight=0)

                G.add_edge("s", f"v'{label_i}", capacity=size - LB, weight=0)
                G.add_edge(f"v'{label_i}", "t", capacity=size - LB, weight=0)

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

        if demand_neg <= demand_pos:
            G.add_node("s", demand=demand_neg - demand_pos)
            G.add_node("t", demand=0)
        else:
            G.add_node("t", demand=demand_neg - demand_pos)
            G.add_node("s", demand=0)

        dem = nx.get_node_attributes(G, "demand", default=None)
        total = 0
        for key in dem:
            total += dem[key]

        min_cost_flow = nx.min_cost_flow(G)

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

    def establish_bound_feasibility_v2(
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

        G.add_edge("t_l", "t", capacity=LB_sum, weilght=0)
        G.add_edge("t_uml", "t", capacity=self.n - LB_sum, weight=0)

        # for e in G.edges:
        #    print(f"Added edge {e} with parameters {G.get_edge_data(u=e[0], v=e[1])}")

        min_cost_flow = nx.max_flow_min_cost(G, "s", "t")

        # print(min_cost_flow)
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
        flow_value = nx.cost_of_flow(G, flow)

        # transform flow into cluster bounds assignment
        cluster_bounds = {label: Tuple[int, int] for label in cluster_sizes}
        for label in cluster_sizes:
            check = False
            size = cluster_sizes[label]
            for j in range(self.k):
                if flow[f"v{label}"][f"w{j}"] == 1:
                    check = True
                    if size < self.LB[j]:
                        cluster_bounds[label] = [self.LB[j], self.UB[j]]
                    elif size > self.UB[j]:
                        cluster_bounds[label] = [self.LB[j], self.UB[j]]
                    else:
                        cluster_bounds[label] = [self.LB[j], self.UB[j]]
            if check is False:
                raise ValueError("Computed invalid flow.")
        if flow_value == 0:
            return cluster_bounds, True

        return cluster_bounds, False

    def establish_bounds(
        self, kmeans_labels: np.ndarray, kmeans_centers: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        # compute assignemnt of cluster sizes to bounds via a min cost flow
        for _ in range(10):
            kmeans_centers = self.compute_centroids(kmeans_labels)
            kmeans_labels = self.compute_assignment(kmeans_centers)
            cluster_sizes = self.kmeans_cluster_sizes(kmeans_labels)

            if len(cluster_sizes) < self.k:
                logger.info("KMeans solution with less than k clusters, skipping...")
                cost_bounded = self.kmeans_cost(kmeans_labels)
                return (cost_bounded, kmeans_labels)

            cluster_bounds, feasible = self.check_bound_feasibility(cluster_sizes)
            if feasible:
                cost_bounded = self.kmeans_cost(kmeans_labels)
                return (cost_bounded, kmeans_labels)

            # if bounds could not be satisfied, establish bounds via min cost flow
            if self.version == "v1":
                self.establish_bound_feasibility_v1(
                    kmeans_labels, kmeans_centers, cluster_sizes, cluster_bounds
                )

            if self.version == "v2":
                self.establish_bound_feasibility_v2(
                    kmeans_labels, kmeans_centers, cluster_sizes, cluster_bounds
                )

        self.sanity_check(kmeans_labels)
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
        if self.LB == [0] * self.k and self.UB == [np.inf] * self.k:
            kmeans_vanilla = KMeans_vanilla(
                n_clusters=self.k, kmeans_iterations=self.kmeans_iterations
            )
            kmeans_vanilla.fit(self.X)
            self.best_inertia = kmeans_vanilla.best_inertia
            self.best_labels = kmeans_vanilla.best_labels

        # if bounds provided run vanilla kmeans++ and modify solution until bounds are satisfied
        else:
            for i in range(self.kmeans_iterations):
                kmeans = KMeans(
                    n_clusters=self.k, n_init="auto", init="k-means++", random_state=i
                )
                kmeans.fit(self.X)

                if kmeans.inertia_ >= self.best_inertia:
                    continue

                cost_bounded, labels_bounded = self.establish_bounds(
                    kmeans.labels_, kmeans.cluster_centers_
                )

                if cost_bounded < self.best_inertia:
                    self.best_inertia = cost_bounded
                    self.best_labels = labels_bounded

        return self
