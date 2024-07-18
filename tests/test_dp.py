import math
import random
import unittest
from typing import Dict

import numpy as np

from exact_kmeans.dynamic_program import compute_bounds


def test_correctness(n: int, k: int, lb: Dict[int, float]) -> bool:
    r = compute_bounds(n, k, lb)
    opt = bruteforce(n, k, lb)

    return bool((opt == r).all()) if r is not None else False


def bruteforce(n: int, k: int, lb: Dict[int, float]) -> np.ndarray:
    opt = np.full((n + 1, k + 1), np.inf)
    opt[0][0] = 0
    bruteforce_enum(n, k, 0, 0, opt, lb)
    # print(opt)
    return opt


def bruteforce_enum(
    n: int, k: int, n_prev: int, k_prev: int, opt: np.ndarray, lb: Dict[int, float]
) -> None:
    for i in range(1, n + 1):
        n_new = n_prev + i
        k_new = k_prev + 1
        cost_new = opt[n_prev][k_prev] + lb[i]
        if n_new <= n and k_new <= k:
            if opt[n_new][k_new] > cost_new:
                opt[n_new][k_new] = cost_new
                bruteforce_enum(n, k, n_new, k_new, opt, lb)


class TestDynamicProgram(unittest.TestCase):
    N_MIN = 10
    N_MAX = 50
    K_MIN = 2
    SLOPE = 10
    OFFSET = 5

    def test_random(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    lb = {i: 0.0 for i in range(0, n + 1)}
                    for i in range(2, n + 1):
                        lb[i] = random.uniform(lb[i - 1], 1.6 * (lb[i - 1] + 1))

                    # print(lb)
                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Random test with n={n}, k={k} failed.\n"
                            f"Lower bounds: {lb}"
                        ),
                    )

    def test_linear_increase(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend(
                        [self.SLOPE * (i - 1) + self.OFFSET for i in range(2, n + 1)]
                    )
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Linear test with n={n}, k={k}, "
                            f"slope={self.SLOPE}, offset={self.OFFSET} failed.\n"
                            f"Lower bounds: {lb}"
                        ),
                    )

    def test_exponential_increase(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend([2**i for i in range(n - 1)])
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Exponential test with n={n}, k={k} failed.\n"
                            f"Lower bounds: {lb}"
                        ),
                    )

    def test_logarithmic_increase(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend([math.log(i, 2) + 1 for i in range(2, n + 1)])
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (f"Log test with n={n}, k={k} failed.\n" f"Lower bounds: {lb}"),
                    )

    def test_linear_exponential(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend(
                        [
                            self.SLOPE * (i - 1) + self.OFFSET
                            for i in range(2, (n + 1) // 2)
                        ]
                    )
                    values.extend([2**i for i in range((n + 1) // 2, n + 1)])
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Linear and exponential test with n={n}, k={k} failed."
                            f"\nLower bounds: {lb}"
                        ),
                    )

    def test_exponential_linear(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend([2**i for i in range(2, (n + 1) // 2)])
                    values.extend(
                        [
                            self.SLOPE * (i - 1) + self.OFFSET
                            for i in range((n + 1) // 2, n + 1)
                        ]
                    )
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Exponential and linear test with n={n}, k={k} failed."
                            f"\nLower bounds: {lb}"
                        ),
                    )

    def test_exponential_logarithmic(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend([2**i for i in range(2, (n + 1) // 2)])
                    values.extend(
                        [math.log(i, 2) + 1 for i in range((n + 1) // 2, n + 1)]
                    )
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Exponential and log test with n={n}, k={k} failed."
                            f"\nLower bounds: {lb}"
                        ),
                    )

    def test_logarithmic_exponential(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend([math.log(i, 2) + 1 for i in range(2, (n + 1) // 2)])
                    values.extend([2**i for i in range((n + 1) // 2, n + 1)])
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Log and exponential test with n={n}, k={k} failed."
                            f"\nLower bounds: {lb}"
                        ),
                    )

    def test_linear_logarithmic(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend(
                        [
                            self.SLOPE * (i - 1) + self.OFFSET
                            for i in range(2, (n + 1) // 2)
                        ]
                    )
                    values.extend(
                        [math.log(i, 2) + 1 for i in range((n + 1) // 2, n + 1)]
                    )
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Linear and log test with n={n}, k={k} failed."
                            f"\nLower bounds: {lb}"
                        ),
                    )

    def test_logarithmic_linear(self) -> None:
        for n in range(self.N_MIN, self.N_MAX):
            for k in range(self.K_MIN, n // 2):
                with self.subTest(msg=f"n={n}, k={k}"):
                    values = [0.0, 0.0]
                    values.extend([math.log(i, 2) + 1 for i in range(2, (n + 1) // 2)])
                    values.extend(
                        [
                            self.SLOPE * (i - 1) + self.OFFSET
                            for i in range((n + 1) // 2, n + 1)
                        ]
                    )
                    lb = {i: values[i] for i in range(0, n + 1)}

                    self.assertTrue(
                        test_correctness(n, k, lb),
                        (
                            f"Log and linear test with n={n}, k={k} failed."
                            f"\nLower bounds: {lb}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
