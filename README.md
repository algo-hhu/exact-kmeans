# K-Means-ILP

1. [Install poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
2. Install the dependencies
```bash
poetry install
```
3. Run the program
```bash
poetry run python -m kmeans_ilp --use-kmeanspp-solution --data-path test_30.csv --verbose --results-path test/test.json --k 3
```

Your `data-path` should be a file containing only the comma-separated data points.
