import matplotlib.pyplot as plt
import numpy as np
import time
from value_estimator import MonteCarloDiscrete, TDLambdaDiscrete, MonteCarloContinuous
from inverted_pendulum import InvertedPendulum
from environment import Actions, Gridworld
from control_method import QLearning, ReinforceSoftmax, QLearningPendulum


# Create pendulum environment and instance of QLearningPendulum
# 0 < control_timestep <= sim_timestep
def initialize_env_and_solver(sim_timestep, control_timestep):
    params = {"mass": 1, "length": 1, "gravity": 1, "damping": 0.05}
    initial_state = np.array([np.pi + 0.1, 0])
    env = InvertedPendulum(
        params=params,
        initial_state=initial_state,
        sim_timestep=sim_timestep,
        control_timestep=control_timestep,
    )
    solver = QLearningPendulum(
        env,
        initial_state=initial_state,
        epsilon=0.2,
        discount_factor=0.9,
        learning_rate=1e-3,
        max_timestep=100,
    )
    return solver


# Run episodes of Q-learning control method, return weights of state-value-function approximation
# num_episodes > 0
def get_approx_weights(solver, num_episodes):
    start = time.time_ns()
    print("Running Q-Learning value function approximation")
    for i in range(n := num_episodes):
        prev_weight_vector = solver.weights
        solver.initial_state = np.array([np.pi + 0.1 * np.random.random(), 0])
        solver.run_episode()
        weight_update_size = np.linalg.norm(
            np.subtract(solver.weights, prev_weight_vector)
        )
        print(
            f"Running episode {solver.episode_count}, magnitude of weight update is {weight_update_size}"
        )
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")
    return solver.weights


# Return res x res matrix (representing true value function) running sample episodes to sample true value function
def get_true_value_func_samples(solver, res, pos_min, pos_max, vel_min, vel_max):
    env = solver.env
    pos_space = np.linspace(pos_min, pos_max, res)
    vel_space = np.linspace(vel_min, vel_max, res)
    true_samples = np.zeros((res, res))

    for pos_i in range(res):
        for vel_i in range(res):
            state_traj = np.zeros(
                (
                    solver.initial_state.size,
                    int(solver.max_timestep // env.control_timestep) + 1,
                )
            )
            state_traj[:, 0] = np.array([pos_space[pos_i], vel_space[vel_i]])
            rewards = np.zeros(int(solver.max_timestep // env.control_timestep))
            for t in range(int(solver.max_timestep // env.control_timestep)):
                next_action = solver.get_epsilon_greedy_action(
                    state_traj[:, t], epsilon=0
                )
                state_traj[:, t + 1] = env.get_next_state(next_action, state_traj[:, t])
                rewards[t] = env.get_state_reward(state_traj[:, t])
            state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)

            return_i = rewards[-1]
            for i in range(rewards.shape[0] - 1):
                return_i *= solver.discount_factor
                return_i += rewards[rewards.shape[0] - i - 1]
            true_samples[vel_i, pos_i] = (
                return_i  # Note: this is so `pos` is on x-axis and `vel` on y-axis
            )

    return true_samples


# Normalize both approximation and true value function, and plot them and generate statistics (TBD)
def compare_value_funcs(approx_weights, true_weights, solver):
    # `approx_weights` and `true_weights` are weights of the action-value func approximation
    # Returns the Euclidean distance between a vector of samples of the two approximations over state-space,
    # after normalizing sample to [0,1]

    # Sample range
    pos_min = -1 * np.pi
    pos_max = np.pi
    vel_min = -1
    vel_max = 1
    torque_min = -1
    torque_max = 1
    res_pos = 20  # Sampling resolution along each axis
    res_vel = 20
    res_torque = 5
    approx_q_samples = np.zeros((res_pos, res_vel))
    true_q_samples = np.zeros((res_pos, res_vel))

    pos_space = np.linspace(pos_min, pos_max, res_pos)
    vel_space = np.linspace(vel_min, vel_max, res_vel)
    torque_space = np.linspace(torque_min, torque_max, res_torque)

    for i in range(pos_space.size):
        for j in range(vel_space.size):
            state = np.array([pos_space[i], vel_space[j]])
            sample_q_vals = np.zeros(res_torque)
            for k in range(res_torque):
                sample_q_vals[k] = solver.action_value_approx(torque_space[k], state)
            best_torque = torque_space[np.argmax(sample_q_vals)]

            approx_q_samples[i, j] = np.dot(
                approx_weights, solver.env.action_feature_vector(best_torque, state)
            )
            true_q_samples[i, j] = np.dot(
                true_weights, solver.env.feature_vector(state)
            )

    approx_min = np.min(approx_q_samples)
    approx_max = np.max(approx_q_samples)
    true_min = np.min(true_q_samples)
    true_max = np.max(true_q_samples)

    approx_q_samples = (approx_q_samples - [approx_min]) / ([approx_max - approx_min])
    true_q_samples = (true_q_samples - [true_min]) / ([true_max - true_min])

    approx_q_samples = np.reshape(approx_q_samples, shape=(-1,))
    true_q_samples = np.reshape(true_q_samples, shape=(-1,))

    return np.linalg.norm(approx_q_samples - true_q_samples)


# Plot both approx and true value function
def plot_value_funcs(solver, approx_weights, true_value_samples, res):
    fig, axs = plt.subplots(1, 2)

    initial_state = solver.initial_state

    env = InvertedPendulum(
        params=solver.env.params,
        initial_state=initial_state,
        sim_timestep=solver.env.sim_timestep,
        control_timestep=solver.env.control_timestep,
    )

    x_min = 0  # Limits of sampled state-space (x=pos, y=vel)
    x_max = 2 * np.pi
    y_min = -1
    y_max = 1

    img = axs[0].imshow(
        true_value_samples,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto",
    )
    fig.colorbar(img, ax=axs[0])
    axs[0].set_aspect((x_max - x_min) / (y_max - y_min))
    axs[0].set_xlabel("pos")
    axs[0].set_ylabel("vel")
    axs[0].set_title("True value function")

    solver.weights = approx_weights

    max_torque = env.params["mass"] * env.params["gravity"] * env.params["length"]
    X = np.linspace(x_min, x_max, res)  # State-action-space coordinates
    Y = np.linspace(y_min, y_max, res)
    max_q_vals = np.zeros((res, res))
    max_q_actions = np.zeros((res, res))
    T = np.linspace(-1 * max_torque, max_torque, 11)
    arr_x = 0  # Array coordinates
    arr_y = 0
    for x in X:
        for y in Y:
            sample_qs_at_x_y = np.zeros(T.size)
            for i in range(T.size):
                sample_qs_at_x_y = solver.action_value_approx(T[i], (x, y))
            max_q_vals[arr_y, arr_x] = np.max(sample_qs_at_x_y)
            max_q_actions[arr_y, arr_x] = np.argmax(sample_qs_at_x_y)
            arr_y += 1
        arr_y = 0
        arr_x += 1

    img = axs[1].imshow(max_q_vals, origin="lower", extent=[x_min, x_max, y_min, y_max])
    axs[1].set_title("Approximated value function\n(Maximum Q-value for each state)")
    fig.colorbar(img, ax=axs[1])
    axs[1].set_aspect((x_max - x_min) / (y_max - y_min))
    axs[1].set_xlabel("pos")
    axs[1].set_ylabel("vel")

    sim_time = solver.max_timestep
    sim_timestep = solver.env.sim_timestep
    state_traj = np.zeros((initial_state.size, int(sim_time // sim_timestep) + 1))
    state_traj[:, 0] = initial_state
    solver.state = solver.initial_state
    for i in range(int(sim_time // sim_timestep)):
        # set first argument to 0 to test without control
        next_state = env.get_next_state(
            solver.get_epsilon_greedy_action(solver.state, epsilon=0), state_traj[:, i]
        )
        state_traj[:, i + 1] = next_state
        solver.state = next_state
    state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)

    plt.figure(2)
    plt.scatter(
        state_traj[0, :], state_traj[1, :], c=range(int(sim_time // sim_timestep) + 1)
    )
    plt.title("State space")
    plt.colorbar(label="Timestep")
    plt.xlabel("pos")
    plt.ylabel("vel")

    plt.show()


solver = initialize_env_and_solver(sim_timestep=0.01, control_timestep=0.1)
approx_weights = get_approx_weights(solver, num_episodes=100)
true_weights = get_true_value_func_samples(
    solver=solver, res=10, pos_min=0, pos_max=2 * np.pi, vel_min=-1, vel_max=1
)
plot_value_funcs(solver, approx_weights, true_weights, res=10)
