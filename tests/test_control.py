import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import control


class TestLQR(unittest.TestCase):
    def test_feedback_from_p_matches_solver(self):
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.diag([10.0, 1.0])
        R = np.array([[0.5]])

        K, P = control.solve_discrete_LQR(
            A, B, Q, R, tolerance=1e-12, max_iterations=10_000
        )
        K_from_P = control.feedback_from_P(P, A, B, R)

        np.testing.assert_allclose(K, K_from_P, atol=1e-9)
        self.assertTrue(np.allclose(P, P.T, atol=1e-9))

    def test_scalar_system_matches_closed_form(self):
        A = np.array([[1.0]])
        B = np.array([[1.0]])
        Q = np.array([[1.0]])
        R = np.array([[1.0]])

        K, P = control.solve_discrete_LQR(A, B, Q, R, tolerance=1e-12)

        p_expected = 0.5 * (Q[0, 0] + np.sqrt(Q[0, 0] ** 2 + 4 * Q[0, 0] * R[0, 0]))
        k_expected = p_expected / (R[0, 0] + p_expected)

        np.testing.assert_allclose(P, np.array([[p_expected]]), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(K, np.array([[k_expected]]), rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
