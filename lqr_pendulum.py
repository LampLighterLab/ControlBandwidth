"""Run a continuous-time LQR controller on the inverted pendulum, plot results, and animate it."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from animation import get_animation
from control import solve_continuous_LQR
from inverted_pendulum import InvertedPendulum


def simulate_LQR_pendulum() -> None:
    params = {"mass": 1.0, "length": 1.0, "gravity": 9.81, "damping": 0.05}
    equilibrium_state = np.array([math.pi, 0.0])
    initial_state = np.array([math.pi - 0.4, 0.25])

    env = InvertedPendulum(
        params=params,
        initial_state=equilibrium_state.copy(),
        sim_timestep=0.01,
        control_timestep=0.05,
    )
    A, B = env.linearize(equilibrium_state, 0.0)
    Q = np.diag([25.0, 2.5])
    R = np.array([[0.75]])
    K, _ = solve_continuous_LQR(A, B, Q, R)

    horizon = 240
    states = np.zeros((horizon + 1, initial_state.size))
    torques = np.zeros(horizon)
    states[0] = initial_state.copy()
    current_state = initial_state.copy()

    torque_limit = params["mass"] * params["gravity"] * params["length"]
    for t in range(horizon):
        error = current_state - equilibrium_state
        torque = -K @ error
        torque = np.clip(torque, -torque_limit, torque_limit)
        torques[t] = torque
        current_state = env.get_next_state(torque[0], current_state.copy())
        states[t + 1] = current_state

    time_axis = np.arange(horizon + 1) * env.control_timestep
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(time_axis, states[:, 0])
    axes[0].axhline(equilibrium_state[0], color="k", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Angle [rad]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, states[:, 1], color="tab:orange")
    axes[1].axhline(equilibrium_state[1], color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Angular vel [rad/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].step(time_axis[:-1], torques, where="post", color="tab:red")
    axes[2].set_ylabel("Torque")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "lqr_timeseries.png"
    fig.savefig(plot_path)
    print(f"Saved time-series plot to {plot_path}")

    torque_for_frames = np.concatenate([torques, torques[-1:]])
    anim_path = out_dir / "lqr_animation.mp4"
    get_animation(states=states.T, actions=torque_for_frames, filename=str(anim_path))


if __name__ == "__main__":
    simulate_LQR_pendulum()
