"""Generate LQR-based ground-truth Q samples for training/testing and save to disk."""

import math
from pathlib import Path

import numpy as np

from control import solve_continuous_LQR
from inverted_pendulum import InvertedPendulum


def lqr_gain(env: InvertedPendulum, equilibrium_state: np.ndarray):
    A_c, B_c = env.linearize(equilibrium_state, 0.0)
    Q = np.diag([25.0, 2.5])
    R = np.array([[0.75]])
    K, _ = solve_continuous_LQR(A_c, B_c, Q, R)
    torque_limit = env.params["mass"] * env.params["gravity"] * env.params["length"]
    return K, torque_limit


def lqr_policy(
    K: np.ndarray, eq_state: np.ndarray, torque_limit: float, state: np.ndarray
) -> np.ndarray:
    torque = -(K @ (state - eq_state)).reshape(-1)[0]
    return np.clip(torque, -torque_limit, torque_limit)


def rollout_return(
    env: InvertedPendulum,
    K: np.ndarray,
    eq_state: np.ndarray,
    torque_limit: float,
    start_state: np.ndarray,
    first_torque: float,
    discount: float,
    horizon_steps: int,
) -> float:
    state = start_state.copy()
    torque = first_torque
    rewards = []
    for _ in range(horizon_steps):
        rewards.append(env.get_reward(torque, state))
        state = env.get_next_state(torque, state)
        torque = lqr_policy(K, eq_state, torque_limit, state)

    ret = 0.0
    for r in reversed(rewards):
        ret = r + discount * ret
    return ret


def sample_true_q(
    env: InvertedPendulum,
    K: np.ndarray,
    eq_state: np.ndarray,
    torque_limit: float,
    pos_range: np.ndarray,
    vel_range: np.ndarray,
    torque_range: np.ndarray,
    discount: float,
    max_sim_time: float,
) -> np.ndarray:
    horizon_steps = int(max_sim_time // env.control_timestep)
    res_pos = pos_range.size
    res_vel = vel_range.size
    res_torque = torque_range.size
    samples = np.zeros((res_pos, res_vel, res_torque))
    for i, pos in enumerate(pos_range):
        for j, vel in enumerate(vel_range):
            state = np.array([pos, vel])
            for k, torque in enumerate(torque_range):
                samples[i, j, k] = rollout_return(
                    env,
                    K,
                    eq_state,
                    torque_limit,
                    state,
                    first_torque=torque,
                    discount=discount,
                    horizon_steps=horizon_steps,
                )
    return samples


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    params = {"mass": 1.0, "length": 1.0, "gravity": 9.81, "damping": 0.05}
    equilibrium_state = np.array([math.pi, 0.0])
    env = InvertedPendulum(
        params=params,
        initial_state=equilibrium_state.copy(),
        sim_timestep=0.01,
        control_timestep=0.05,
    )

    K, torque_limit = lqr_gain(env, equilibrium_state)
    max_sim_time = 10.0
    discount = 0.98

    pos_range = np.linspace(0.0, 2 * math.pi, 25)
    vel_range = np.linspace(-3.5, 3.5, 25)
    torque_range = np.linspace(-torque_limit, torque_limit, 13)

    print("Sampling LQR ground truth on a single grid...")
    q = sample_true_q(
        env,
        K,
        equilibrium_state,
        torque_limit,
        pos_range,
        vel_range,
        torque_range,
        discount=discount,
        max_sim_time=max_sim_time,
    )

    save_path = data_dir / "lqr_ground_truth.npz"
    np.savez(
        save_path,
        params=params,
        equilibrium_state=equilibrium_state,
        discount=discount,
        max_sim_time=max_sim_time,
        pos_range=pos_range,
        vel_range=vel_range,
        torque_range=torque_range,
        q=q,
    )
    print(f"Saved ground-truth rollouts to {save_path}")


if __name__ == "__main__":
    main()
