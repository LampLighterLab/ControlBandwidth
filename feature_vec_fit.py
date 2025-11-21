from inverted_pendulum import InvertedPendulum
import numpy as np
import itertools
from util import sample
from control_method import QLearningPendulum
from run_pendulum_ctrl_estim import run_q_learning
import pickle

class InvertedPendulumFit(InvertedPendulum):

    def __init__(self, params, initial_state, sim_timestep=0.01, control_timestep=0.1):
        super().__init__(params, initial_state, sim_timestep, control_timestep)
        self.action_feature_vector_length = super().action_feature_vector(0, [0, 0]).shape[0]
        self.mask = np.ones(self.action_feature_vector_length)

    # `mask` is vector of 0s and 1s
    def action_feature_vector(self, a, s):
        return np.multiply(super().action_feature_vector(a, s), self.mask)  # elementwise product

# just the part that gets the value samples from util.py
def get_true_value_samples(solver, res, t_res, x_min, x_max, y_min, y_max):
    env = solver.env
    torque_range = np.linspace(-1*solver.max_torque, solver.max_torque, t_res)
    # sample_matrix[pos, vel, torque]
    sample_q_matrix = np.zeros((res, res, t_res))

    def sample_q(pos, vel, torque):
        print(f"sampling pos {pos:.3f}, vel {vel:.3f}, torque {torque:.3f}")
        num_timesteps = int(solver.max_sim_time // env.control_timestep)
        state_traj = np.zeros(
            (
                solver.initial_state.size,
                num_timesteps + 1,
            )
        )
        state_traj[:, 0] = np.array([pos, vel])
        rewards = np.zeros(num_timesteps)

        # Take first action
        next_action = torque
        state_traj[:, 1] = env.get_next_state(next_action, state_traj[:, 0])
        rewards[0] = env.get_reward(next_action, state_traj[:, 0])
        # Follow policy
        for t in range(1, num_timesteps):
            next_action = solver.get_epsilon_greedy_action(state_traj[:, t], epsilon=0)
            state_traj[:, t + 1] = env.get_next_state(next_action, state_traj[:, t])
            rewards[t] = env.get_reward(next_action, state_traj[:, t])
        state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)

        return_i = rewards[-1]
        for i in range(rewards.shape[0] - 1):
            return_i *= solver.discount_factor
            return_i += rewards[rewards.shape[0] - i - 1]
        return return_i

    # Fill out sample_q_matrix
    for torque_i in range(t_res):

        def sample_q_container(pos, vel):
            return sample_q(pos, vel, torque=torque_range[torque_i])

        sample_q_matrix[:, :, torque_i] = sample(
            func=sample_q_container,
            res=res,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )
    sample_q_matrix = np.transpose(
        sample_q_matrix, axes=(1, 0, 2)
    )  # sample() flips axes 0,1

    # normalize to [0, 1]
    min = np.min(sample_q_matrix)
    max = np.max(sample_q_matrix)
    sample_q_matrix = (sample_q_matrix - min) / (max - min)
    return sample_q_matrix

# Benchmark different combinations of features against ground truth using least squares
# matrix `true_value_samples` should match res, t_res, x_min, etc.
def benchmark(solver, true_value_samples, res, t_res, x_min, x_max, y_min, y_max):
    env = solver.env
    pos_range = np.linspace(x_min, x_max, res)
    vel_range = np.linspace(y_min, y_max, res)
    feature_vec_len = env.action_feature_vector(0, np.array([0, 0])).shape[0]
    torque_range = np.linspace(-1*solver.max_torque, solver.max_torque, t_res)

    # masks[i, :] is mask i
    masks = np.array(list(itertools.product([0, 1], repeat=feature_vec_len)))
    least_squares_weights = np.zeros_like(masks)

    errors = np.zeros((masks.shape[0]))

    for i, mask in enumerate(masks):
        solver.env.mask = mask
        X = np.zeros((res, res, t_res, feature_vec_len))
        for pos_i in range(res):
            for vel_i in range(res):
                for torque_i in range(t_res):
                    torque = torque_range[torque_i]
                    state = np.array([pos_range[pos_i], vel_range[vel_i]])
                    X[pos_i, vel_i, torque_i, :] = env.action_feature_vector(torque, state)
        X = X.reshape((-1, feature_vec_len))
        Y = true_value_samples.reshape(-1)
        least_squares_weights, _, _, _ = np.linalg.lstsq(X, Y)

        # Generate sample Q matrix
        solver.weights = least_squares_weights
        approx_value_samples = np.zeros((res, res, t_res))
        def sample_q(pos, vel, torque):
            return solver.action_value_approx(torque, [pos, vel])

        for torque_i in range(t_res):
            def sample_q_container(pos, vel):
                return sample_q(pos, vel, torque=torque_range[torque_i])
            approx_value_samples[:, :, torque_i] = sample(
                func=sample_q_container,
                res=res,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
            )
        approx_value_samples = np.transpose(
            approx_value_samples, axes=(1, 0, 2)
        )  # sample() flips axes 0,1
        # normalize to [0, 1]
        min = np.min(approx_value_samples)
        max = np.max(approx_value_samples)
        approx_value_samples = (approx_value_samples - min) / (max - min)

        errors[i] = np.linalg.norm((true_value_samples - approx_value_samples))
        print(f"Mask {i} has error {errors[i]:.5f}, mask is {mask}")

    return masks, errors


params = {"mass": 1, "length": 1, "gravity": 1, "damping": 0.01}
initial_state = np.array([np.pi + 0.1, 0])
env = InvertedPendulumFit(
    params=params,
    initial_state=initial_state,
    sim_timestep=0.01,
    control_timestep=0.05,
)
solver = QLearningPendulum(
    env,
    initial_state=initial_state,
    epsilon=0.2,
    discount_factor=0.98,
    learning_rate=1e-4,
    max_sim_time=10,
)

# save true value samples to file
'''
run_q_learning(solver=solver, num_episodes=200)
true_value_samples = get_true_value_samples(solver=solver, res=21, t_res=7, x_min=0, x_max=2*np.pi, y_min=-3, y_max=3)
with open("true_value_samples.pkl", "wb") as file:
    pickle.dump(true_value_samples, file)
    print("true value samples saved")
'''

with open("true_value_samples.pkl", "rb") as file:
    true_value_samples = pickle.load(file)

masks, errors = benchmark(solver, true_value_samples, res=21, t_res=7, x_min=0, x_max=2*np.pi, y_min=-3, y_max=3)
argmin_error = np.argmin(errors[1:])
best_mask = masks[argmin_error, :]
print(f"Best mask was {best_mask} with error {errors[argmin_error]}")