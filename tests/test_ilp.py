import logging
import sys
import unittest

import numpy as np
import pandas as pd

from exact_kmeans import ExactKMeans

formatter = logging.Formatter(
    "[%(asctime)s: %(levelname)s/%(filename)s:%(lineno)d] %(message)s"
)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stdout_handler)

logging.getLogger().setLevel(logging.INFO)


class TestILP(unittest.TestCase):
    def test_circles(self) -> None:
        X = pd.read_csv("tests/datasets/circlesuneven_4_20_0.csv")
        ilp = ExactKMeans(X=X, k=4)
        res = ilp.optimize()

        print("Found objective value:", res["objective"])

        assert np.isclose(
            res["objective"], 1276.8469883712723
        ), f"The found objective value is wrong: {res['objective']}"


if __name__ == "__main__":
    unittest.main()
