import matplotlib.pyplot as plt
import numpy as np


# Helper function
# Returns a 2D nparray sampling the function `func` with arr[y, x] = func(x, y) (for plotting purposes)
# within bounds with resolution `res`
def sample(func, res, x_min, x_max, y_min, y_max, dtype=np.float64):
    x_range = np.linspace(x_min, x_max, res)
    y_range = np.linspace(y_min, y_max, res)
    samples = np.zeros((res, res), dtype=dtype)
    for x_i in range(res):
        for y_i in range(res):
            samples[y_i, x_i] = func(x_range[x_i], y_range[y_i])
    return samples


# Generate a sample trajectory using `solver`, return state array, action array
# state_traj[0, i] is position at timestep i, state_traj[1, i] is velocity at timestep i
# action_traj[i] is action taken at timestep i
def sample_trajectory(solver):
    env = solver.env
    initial_state = solver.initial_state
    num_timesteps = int(solver.max_timestep // solver.env.sim_timestep)
    state_traj = np.zeros((initial_state.size, num_timesteps + 1))
    action_traj = np.zeros(num_timesteps + 1)
    state_traj[:, 0] = initial_state
    solver.state = solver.initial_state
    for i in range(num_timesteps):
        action_traj[i] = solver.get_epsilon_greedy_action(solver.state, epsilon=0)
        # set first argument to 0 to test without control
        next_state = env.get_next_state(action_traj[i], state_traj[:, i])
        state_traj[:, i + 1] = next_state
        solver.state = next_state
    state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)
    return state_traj, action_traj


# Compute least-squares solution of feature vector weights for testing
# Samples ground truth over state-action-space in a res x res x t_res grid
# pos on [x_min, x_max], vel on [y_min, y_max], torque on [t_min, t_max]
def least_squares_fit(solver, res, x_min, x_max, y_min, y_max, t_min, t_max):
    env = solver.env
    t_res = 5
    pos_range = np.linspace(x_min, x_max, res)
    vel_range = np.linspace(y_min, y_max, res)
    torque_range = np.linspace(t_min, t_max, t_res)
    # sample_matrix[pos, vel, torque]
    sample_q_matrix = np.zeros((res, res, t_res))

    def sample_q(pos, vel, torque):
        num_timesteps = int(solver.max_timestep // env.control_timestep)
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

    # Fill out sample_matrix
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

    # Compute least squares regression
    feature_vec_len = env.action_feature_vector(0, np.array([0, 0])).shape[0]
    X = np.zeros((res, res, t_res, feature_vec_len))
    for pos_i in range(res):
        for vel_i in range(res):
            for torque_i in range(t_res):
                torque = torque_range[torque_i]
                state = np.array([pos_range[pos_i], vel_range[vel_i]])
                X[pos_i, vel_i, torque_i, :] = env.action_feature_vector(torque, state)

    X = X.reshape((-1, feature_vec_len))
    Y = sample_q_matrix.reshape(-1)
    feature_vec_coeffs = (
        np.linalg.inv((X.T @ X) + 1e-10 * np.identity(feature_vec_len)) @ X.T @ Y
    )
    return feature_vec_coeffs
