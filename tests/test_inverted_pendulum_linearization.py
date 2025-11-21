import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inverted_pendulum import InvertedPendulum


VISUALIZE_VECTOR_FIELD = False


class InvertedPendulumLinearizationTest(unittest.TestCase):
    def setUp(self):
        params = {"mass": 1.0, "length": 1.0, "gravity": 9.81, "damping": 0.05}
        initial_state = np.array([0.0, 0.0])
        self.env = InvertedPendulum(
            params=params,
            initial_state=initial_state,
            sim_timestep=0.001,
            control_timestep=0.005,
        )
        self.rng = np.random.default_rng(2024)
        self.max_torque = (
            0.5
            * self.env.params["mass"]
            * self.env.params["gravity"]
            * self.env.params["length"]
        )

    def test_linearization_matches_finite_difference(self):
        eps_state = 1e-6
        eps_control = 1e-5
        state_bounds = np.array([[-np.pi, np.pi], [-2.0, 2.0]])

        samples = []
        for _ in range(8):
            state = np.array(
                [
                    self.rng.uniform(state_bounds[0, 0], state_bounds[0, 1]),
                    self.rng.uniform(state_bounds[1, 0], state_bounds[1, 1]),
                ]
            )
            torque = self.rng.uniform(-self.max_torque, self.max_torque)

            A_lin, B_lin = self.env.linearize(state, torque)
            numeric_A = np.zeros_like(A_lin)

            for i in range(state.size):
                delta = np.zeros_like(state)
                delta[i] = eps_state
                forward = self.env.compute_dynamics(state + delta, torque)
                backward = self.env.compute_dynamics(state - delta, torque)
                numeric_A[:, i] = (forward - backward) / (2.0 * eps_state)

            forward_control = self.env.compute_dynamics(state, torque + eps_control)
            backward_control = self.env.compute_dynamics(state, torque - eps_control)
            numeric_B = (
                (forward_control - backward_control) / (2.0 * eps_control)
            ).reshape(-1, 1)

            np.testing.assert_allclose(
                A_lin,
                numeric_A,
                rtol=2e-3,
                atol=5e-5,
                err_msg=f"Linearization A mismatch at state {state}, torque {torque}",
            )
            np.testing.assert_allclose(
                B_lin,
                numeric_B,
                rtol=2e-3,
                atol=5e-5,
                err_msg=f"Linearization B mismatch at state {state}, torque {torque}",
            )

            flow_analytic = self.env.compute_dynamics(state, torque)
            next_state = self.env.get_next_state(torque, state.copy())
            flow_numeric = (next_state - state) / self.env.control_timestep

            samples.append(
                {
                    "state": state,
                    "torque": torque,
                    "A_lin": A_lin,
                    "B_lin": B_lin,
                    "A_num": numeric_A,
                    "B_num": numeric_B,
                    "flow_analytic": flow_analytic,
                    "flow_numeric": flow_numeric,
                }
            )

        if VISUALIZE_VECTOR_FIELD:
            self._visualize_vector_field(samples)

    def _visualize_vector_field(self, samples):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError as exc:  # pragma: no cover - optional path
            raise unittest.SkipTest("matplotlib required for visualization") from exc

        states = np.array([sample["state"] for sample in samples])
        analytic_flows = np.array([sample["flow_analytic"] for sample in samples])
        numeric_flows = np.array([sample["flow_numeric"] for sample in samples])
        flow_errors = np.linalg.norm(analytic_flows - numeric_flows, axis=1)

        offset = 0.03

        plt.figure(figsize=(8, 4))
        plt.quiver(
            states[:, 0],
            states[:, 1],
            analytic_flows[:, 0],
            analytic_flows[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.007,
            color="tab:blue",
        )
        plt.quiver(
            states[:, 0] + offset,
            states[:, 1] + offset,
            numeric_flows[:, 0],
            numeric_flows[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.007,
            color="tab:orange",
        )

        legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=2, label="Analytic flow"),
            Line2D([0], [0], color="tab:orange", lw=2, label="Numeric flow"),
        ]
        plt.legend(handles=legend_handles, loc="upper right")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Angular velocity [rad/s]")
        plt.title("Linearized Flow at Sampled Points")
        plt.tight_layout()

        plt.figure(figsize=(7, 3))
        scatter = plt.scatter(
            states[:, 0],
            states[:, 1],
            c=flow_errors,
            cmap="inferno",
            s=80,
            edgecolor="k",
        )
        plt.colorbar(scatter, label="‖Analytic flow - Numeric flow‖₂")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Angular velocity [rad/s]")
        plt.title("Flow Discrepancy Magnitude")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
