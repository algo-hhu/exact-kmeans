import logging
import unittest

import pandas as pd

from exact_kmeans import ExactKMeans

logging.getLogger().setLevel(logging.INFO)


class TestILP(unittest.TestCase):
    def test_circles(self) -> None:
        X = pd.read_csv("tests/datasets/circlesuneven_4_20_0.csv")
        ilp = ExactKMeans(X=X.values, k=4)
        res = ilp.optimize()

        print(res)


if __name__ == "__main__":
    unittest.main()
