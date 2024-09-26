[![Build Status](https://github.com/algo-hhu/exact-kmeans/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/exact-kmeans/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Supported Python version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Stable Version](https://img.shields.io/pypi/v/exact-kmeans?label=stable)](https://pypi.org/project/exact-kmeans/)

# exact-kmeans

This package computes exact solutions to the $k$-means problem using integer linear programming (ILP). It can also compute exact solution for k-means with lower and upper bound constraints, outliers and the combination of both.
Since our plain ILP solution to the problem was too slow, we implemented a branch-and-bound algorithm that computes multiple ILP solutions for different cluster sizes to find the optimal cluster sizes $[c_1, \ldots, c_k]$.
The ILPs are implemented using [Gurobi](https://www.gurobi.com/). You will need a [Gurobi license](https://www.gurobi.com/downloads/) to be able to run the code.

The algorithm works as follows:
- We run $k$-means++ on the data to get an initial upper bound for the cost of an optimal solution.
- We try to find the largest cluster size $c_1$ that a cluster in the optimal solution may have. We use the fact that the cost of a cluster in the optimal solution is less than the initial upper bound. We do this by solving an ILP (ILP#1).
- We iterate through all possible cluster sizes $c_1 \geq c_2 \ldots \geq c_k$ and compute for each of them a solution using an ILP (ILP#2). However, we don't actually iterate through all the sizes, but use a branch-and-bound algorithm to skip some of them if the cost exceeds the initial upper bound. You can see an example of this in the tree in the "Plot the Branch and Bound Tree" section.

Both ILPs get cluster sizes $c_1,\ldots, c_i$ with $i\leq k$ as input and compute an optimal solution to the following modification of the $k$-means problem:
- the number of clusters is $i$
- the clusters have sizes $c_1,\ldots, c_i$
- if $n$ is the number of data points $n-c_1-\ldots -c_i$ points are not part of any cluster

ILP#1 gets one single cluster size $c_1$ as input and computes an optimal solution with one single cluster of size $c_1$. We use ILP#1 to compute a lower bound to the cost of a cluster of size $c_1$. If this cost is already larger than the initial upper bound, then we know that the cluster sizes of an optimal solution must be all less than $c_1$. Therefore we can discard all cluster sizes greater equal $c_1$ when searching for the cluster sizes of an optimal solution.

ILP#2 gets cluster sizes $c_1,\ldots, c_i$ with $2\leq i\leq k$ as input. If $c_1,\ldots, c_i$ are also cluster sizes which are represented in the optimal solution the cost that ILP#2 computes should be less than the initial upper bound. We add a constraint to ILP#2 bounding the cost by initial upper bound. Therefore either ILP#2 is infeasible or it outputs an optimal solution of $i$ clusters with sizes $c_1,\ldots, c_i$ and cost smaller than initial upper bound.

To search for the cluster sizes of an optimal solution we use a branch and bound approach. In a branch node of level $i$ the $i$ largest cluster sizes $c_1, \ldots, c_i$ are already fixed. The variables `branching_levels` and `fill_cluster_sizes` define the behavior on these branching nodes. If `branching_levels` is greater equal to the level $i$ of the node then we use ILP#2 to bound the current cost and decide if we branch, otherwise we always branch on this node. If the variable `fill_cluster_sizes` is set to true we compute the smallest possible remaining cluster sizes $c_{i+1},\ldots, c_{k}$ and run ILP#2 with cluster sizes $c_1,\ldots, c_k$. If the variable `fill_cluster_sizes` is set to false we run ILP#2 only with the fixed cluster sizes $c_1,\ldots, c_i$. Setting `fill_cluster_sizes` to true may lead to less branching but can increase the solving time of ILP#2.

## Config file
To customize the runs, you can create a config file. The default config file is [`exact_kmeans/config/default.yaml`](exact_kmeans/config/default.yaml). You can also pass a different config file as an argument.
- `num_processes` (integer or float) sets the number of processes used. The algorithm was parallelized using the `multiprocessing` package, so you can set the number of processes that you want to use. If you use an integer, at most that number of processes will be taken, otherwise if you use a float, it will be a fraction of the available CPUs. If the parameter is not passed, the algorithm will use all available CPUs.
- `branching_priorities` (true/false) is passed to the ILP#2 model and enable/disable the use of branching priorities. If true, the priority of the $x$ variable will be higher than the other variables. According to our tests, this speeds up the solving.
- `warm_start` (true/false) If set to true a pre-processing step is used to find a feasible solution to ILP#2, if it finds a feasible solution the solution is passed to ILP#2. This can decrease the time needed to solve ILP#2.
- `replace_min` (true/false) replaces a minimum constraint with a linear version. According to our tests, this speeds up the solving.
- `branching_levels` (integer) Sets the maximum level of a branching node where we run ILP#2 to bound the cost. Since this can increase the number of ILPs that are computed, it may only make sense for small levels.
- `fill_cluster_sizes` (true/false) If set to false we run ILP#2  only with cluster sizes fixed at the branching node. If set to true we fill the cluster sizes up to $k$. Setting the variable to true can result in a larger computation time for ILP#2 but may result in less branching, which can save time. For small $k$ we recommend setting it to false and for large $k$ we recommend setting it to true.
- `bound_model_params` are the arguments that are passed to the ILP#1 model. Please have a look at the [Gurobi documentation](https://www.gurobi.com/documentation/9.1/refman/parameters.html) for more information.
- `model_params` are the arguments that are passed to the ILP#2 model. Please have a look at the [Gurobi documentation](https://www.gurobi.com/documentation/9.1/refman/parameters.html) for more information.

## Constraints

This package can handle lower bounds, upper bounds and outliers.
- **lower/upper bounds:** for given lower bounds $l_1,..., l_k$ and upper bounds $u_1,..., u_k$  the task is to compute the best clustering with cluster sizes $c_1, ..., c_k$ such that $l_i\leq c_i\leq u_i$ for $i=1,..., k$.
- **outlier:** for a given number of outliers $out$ the task is to compute the best clustering where it is allowed to remove up to $out$ many points from the dataset.
- **lower/upper bounds and outlier:** the task is to compute the best clustering with cluster sizes $c_1, ..., c_k$ such that $l_i\leq c_i\leq u_i$ for $i=1,..., k$ while it is also allowed to remove up to $out$ many points from the dataset.


## Installation

For this package to work, you need to have a working installation of [Gurobi](https://www.gurobi.com/). You can get a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
[Here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer) you can find a guide on how to install it.

```bash
pip install exact-kmeans
```

## Use within Python
```python
class exact_kmeans.ilp.ExactKMeans(n_clusters, config_file: Path(__file__).parent.resolve()/ "config"/ "default.yaml",
  kmeans_iterations = 100, LB = None, UB = None, outlier = 0)
```
### Parameters:
- `n_clusters: int` Number of clusters k of the k-means solution
- `config-file: Union[str, Path], dafault=Path(__file__).parent.resolve()/ "config"/ "default.yaml"` config file for branch and bound and ILP cofigurations, see [here](#config-file)
- `kmeans_iterations: int, default=100` In the unconstrained version this is the number of runs of the k-means++ algortihm, in the constrained version this is the number of runs of the k-means++ algorithm plus a postprocessing step which gurantees that the constrains are satisfied. The solution with smallest cost among all iterations is returned and used as initialization for the branch and bound approach. Better solutions can decrease the computation time of branch and bound.
- `LB: Optional[List], default=None` List of lower bounds for the size of clusters. It must have length `n_clusters`. If provided, please guarantee that `sum(LB)` is smaller equal the size of the data set, otherwise there is no feasible solution.
- `UB: Optional[List], default=None` List of upper bounds for the size of clusters. It must have length `n_clusters`. If provided, please guarantee that `sum(UB)+outlier` is greater equal the size of the data set, otherwise there is no feasible solution.
- `outlier: int, default=0` Number of outliers, it must be non-negative. If provided please gurantee that it is smaller equal the size of the data set. Points labeled as outliers in the computed solution have label `n_clusters`.

## Example:

```python
from exact_kmeans import ExactKMeans
from exact_kmeans.util import JsonEncoder
from ucimlrepo import fetch_ucirepo

# For example, you can import data from the UCI repository
# but you can also use your own dataset
iris = fetch_ucirepo(id=53)

X = iris.data.features

#If desired, add lower bounds LB and upper bounds UB as lists
LB=[24, 20, 45]
UB=[45, 60, 59]

#If desired, add number of outliers
out=10

#vanilla k-means
ilp_1 = ExactKMeans(n_clusters=3)
res_1 = ilp_1.fit(X)

#k-means with bounds and outliers
ilp_2= ExactKMeans(n_clusters=3, LB=LB, UB=UB, outlier=out)
res_2 = ilp_2.fit(X)

with open("output.json", "w") as f:
    json.dump(res_1, f, indent=4, cls=JsonEncoder)

# You can also print the branch and bound tree to visualize the decisions made by the algorithm
# Below you find an example of the tree and meaning of the colors
from exact_kmeans.plot_tree import plot

plot(
  nodes=res_1.processed_cluster_sizes,
  filename="test",
  plot_folder="plots",
  optimal_objective=res_1.model.ObjVal,
)

```

## Command Line

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install the package
```bash
poetry install
```

Run the program for vanilla k-means
```bash
poetry run python -m exact_kmeans --data-path iris.csv --verbose --results-path test/iris.json --k 3 --config-file exact_kmeans/config/default.yaml
```

Run the program for constrained k-means: `bounds-path` and `outlier` are both optional
```bash
poetry run python -m exact_kmeans --data-path iris.csv --verbose --results-path test/iris.json --k 3 --config-file exact_kmeans/config/default.yaml --bounds-path test/iris_bounds.csv --outlier 15
```

Your `data-path` should be a file with a header containing only the comma-separated data points. Example:
```csv
0,1,2,3
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
4.7,3.2,1.3,0.2
4.6,3.1,1.5,0.2
5.0,3.6,1.4,0.2
...
```

Your `bounds-path` should be a file with a header containing comma-separated values. Column with name `LB` contains the lower bounds column with name `UB` contains the upper bounds. If there is no column with name `LB` the lower bounds will be set to the default value 0 if there is no culumn with name `UB` the upper bounds will be set to default value `numpy.inf`. If one of the columns contains less values than the number of clusters k, it will be filled with default values up to k.  Example:

```csv
LB,UB
30,45
12,12
54,112
```

## Plot the Branch and Bound Tree

Assuming that you have run the program and stored the results in a JSON file, you can plot the tree produced by the algorithm.

```bash
poetry run python exact_kmeans.plot_tree.py --output-json output.json --plot-folder plots
```

This will create a tree that looks like this:
<p align="center">
  <img src="https://raw.githubusercontent.com/algo-hhu/exact-kmeans/main/images/tree.png" alt="Tree Example"/>
</p>

The nodes can have the following colors:
- Red: The cluster sizes were infeasible.
- Green: The cluster sizes were feasible.
- Gray: The node was a branching node.
- Orange/Yellow: The node was skipped because it had a too high initial cost.
- The optimal node has a gold border.

The nodes have the following information:
- The cluster sizes that were tested.
- The amount of seconds it took.
- The cost of the solution. This is only shown if the node has $k$ clusters and is feasible.

## Development

Install [poetry](https://python-poetry.org/docs/#installation)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install the package
```bash
poetry install
```

Run the tests
```bash
poetry run python -m unittest discover tests -v
```

## Disclaimer

This package requires the Gurobi Optimizer, which is a commercial optimization solver. Users must obtain their own licenses for Gurobi.
You are responsible for complying with all Gurobi licensing terms. For more information, visit the [Gurobi Licensing Page](https://www.gurobi.com/academia/academic-program-and-licenses/).

<!--
## Citation

If you use this code, please cite [the following paper]():

```
```
-->
