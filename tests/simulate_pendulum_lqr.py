import math
from dataclasses import dataclass

import numpy as np

from control import feedback_from_P, solve_discrete_LQR
from inverted_pendulum import InvertedPendulum


def discretize_linear_system(
    A_c: np.ndarray, B_c: np.ndarray, timestep: float, order: int = 6
):
    """
    Approximate the discrete-time pair (A_d, B_d) for a continuous-time linear system
    using a truncated Taylor expansion of the matrix exponential.

    ẋ = A_c x + B_c u  ->  x[k+1] ≈ A_d x[k] + B_d u[k]
    """
    if order < 1:
        raise ValueError("Order must be >= 1.")

    state_dim = A_c.shape[0]
    A_power = np.eye(state_dim)
    A_d = np.eye(state_dim)
    B_d = np.zeros_like(B_c)

    for k in range(1, order + 1):
        A_power = A_power @ A_c
        coeff = (timestep**k) / math.factorial(k)
        A_d += coeff * A_power

    A_power = np.eye(state_dim)
    for k in range(order):
        coeff = (timestep ** (k + 1)) / math.factorial(k + 1)
        if k > 0:
            A_power = A_power @ A_c
        B_d += coeff * A_power @ B_c

    return A_d, B_d


@dataclass
class SimulationResult:
    states: np.ndarray
    torques: np.ndarray
    gain: np.ndarray
    riccati: np.ndarray


def main():
    params = {"mass": 1.0, "length": 1.0, "gravity": 9.81, "damping": 0.05}
    equilibrium_state = np.array([math.pi, 0.0])
    equilibrium_torque = 0.0

    initial_state = np.array([math.pi - 0.4, 0.2])
    horizon = 200

    env = InvertedPendulum(
        params=params,
        initial_state=equilibrium_state.copy(),
        sim_timestep=0.01,
        control_timestep=0.05,
    )

    A_c, B_c = env.linearize(equilibrium_state, equilibrium_torque)
    A_d, B_d = discretize_linear_system(A_c, B_c, env.control_timestep)

    Q = np.diag([20.0, 2.0])
    R = np.array([[0.5]])

    K, P = solve_discrete_LQR(A_d, B_d, Q, R)
    K_from_P = feedback_from_P(P, A_d, B_d, R)

    states = np.zeros((horizon + 1, initial_state.size))
    torques = np.zeros(horizon)
    states[0] = initial_state.copy()

    current_state = initial_state.copy()
    for t in range(horizon):
        state_error = current_state - equilibrium_state
        torque = -float(K @ state_error)
        torques[t] = torque
        current_state = env.get_next_state(torque, current_state.copy())
        states[t + 1] = current_state

    print("Discrete-time linearization (A_d):")
    print(A_d)
    print("\nDiscrete-time linearization (B_d):")
    print(B_d)
    print("\nSolution to Riccati equation (P):")
    print(P)
    print("\nOptimal feedback gain (K):")
    print(K)
    print("\nGain recovered from P (feedback_from_P):")
    print(K_from_P)
    print("\nFinal state:")
    print(states[-1])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return SimulationResult(states=states, torques=torques, gain=K, riccati=P)

    time_axis = np.arange(horizon + 1) * env.control_timestep

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time_axis, states[:, 0], label="angle [rad]")
    plt.plot(time_axis, states[:, 1], label="angular velocity [rad/s]")
    plt.axhline(equilibrium_state[0], color="k", linestyle="--", linewidth=0.8)
    plt.axhline(equilibrium_state[1], color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("State trajectory")

    plt.subplot(1, 2, 2)
    plt.step(time_axis[:-1], torques, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque")
    plt.title("Control signal")
    plt.tight_layout()
    plt.show()

    return SimulationResult(states=states, torques=torques, gain=K, riccati=P)


if __name__ == "__main__":
    main()
