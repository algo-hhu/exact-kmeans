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
        X = pd.read_csv("tests/datasets/circlesuneven.csv")
        ilp = ExactKMeans(n_clusters=3)
        ilp.fit(X)

        print("Found objective value:", ilp.inertia_)

        assert np.isclose(
            ilp.inertia_, 454.099632
        ), f"The found objective value is wrong: {ilp.inertia_}"


if __name__ == "__main__":
    unittest.main()
