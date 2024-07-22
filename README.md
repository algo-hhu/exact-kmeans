[![Build Status](https://github.com/algo-hhu/exact-kmeans/actions/workflows/mypy-flake-test.yml/badge.svg)](https://github.com/algo-hhu/exact-kmeans/actions)
<!--[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)-->
[![Supported Python version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Stable Version](https://img.shields.io/pypi/v/exact-kmeans?label=stable)](https://pypi.org/project/exact-kmeans/)

# exact-kmeans

This package computes exact solutions to the $k$-means problem using integer linear programming (ILP).
Since our plain ILP solution to the problem was too slow, we implemented a branch-and-bound algorithm that computes multiple ILP solutions for different cluster sizes to find the optimal cluster sizes $[c_1, \ldots, c_k]$.

The algorithm works as follows:
- We run $k$-means++ on the data to get an initial upper bound for the cost.
- We try to find the largest cluster size $c_1$ that any cluster may have and that is less than the initial upper bound. We do this by solving an ILP.
- We iterate through all possible cluster sizes $c_1, \ldots, c_k$ and compute a solution using an ILP. However, we don't actually iterate through all the sizes, but use a branch-and-bound algorithm to skip some of them. You can see an example of this in the tree in the "Plot the Branch and Bound Tree" section.


## Installation

For this package to work, you need to have a working installation of [Gurobi](https://www.gurobi.com/). You can get a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).
[Here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer) you can find a guide on how to install it.

```bash
pip install exact-kmeans
```

## Use within Python

```python
from exact_kmeans import ExactKMeans
from exact_kmeans.util import JsonEncoder
from ucimlrepo import fetch_ucirepo

# For example, you can import data from the UCI repository
# but you can also use your own dataset
iris = fetch_ucirepo(id=53)

X = iris.data.features

km = ExactKMeans(X=X, k=3)
res = ilp.optimize()

with open("output.json", "w") as f:
    json.dump(res, f, indent=4, cls=JsonEncoder)
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

Run the program
```bash
poetry run python -m exact_kmeans --data-path iris.csv --verbose --results-path test/iris.json --k 3 --config-file config/default.yaml
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

## Plot the Branch and Bound Tree

Assuming that you have run the program and stored the results in a JSON file, you can plot the tree produced by the algorithm.

```bash
poetry run python plot_tree.py --output-json output.json --plot-folder plots
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

The nodes have the following information:
- The cluster sizes that were tested.
- The amount of seconds it took.
- The cost of the solution. This is only shown if the node has *k* clusters and is feasible.

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

<!--
## Citation

If you use this code, please cite [the following paper]():

```
```
-->
