import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
import time
from inverted_pendulum import InvertedPendulum
from animation import get_animation
from util import (
    sample,
    sample_trajectory,
    initialize_env_and_solver,
    get_plot_least_squares,
    least_squares_fit
)
import pickle


# Run episodes of Q-learning control method, return weights of state-value-function approximation
# num_episodes > 0
def run_q_learning(solver, num_episodes):
    start = time.time_ns()
    print("Running Q-Learning value function approximation")
    n = num_episodes
    states = np.zeros((2, 0))
    for i in range(n):
        print(solver.weights)
        # Decaying exponential learning rate
        # solver.learning_rate *= 0.997
        prev_weight_vector = solver.weights
        # Random initial state
        random_initial_state = [
            np.pi + 0.25 * np.pi * np.random.random(),
            0.5 * np.random.random(),
        ]
        solver.initial_state = np.array(random_initial_state)
        episode_states = solver.run_episode()
        states = np.append(states, episode_states, axis=1)
        weight_update_size = np.linalg.norm(
            np.subtract(solver.weights, prev_weight_vector)
        )
        print(
            f"Running episode {solver.episode_count}, magnitude of weight update is {weight_update_size}"
        )
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")
    states[0, :] = np.mod(states[0, :], 2 * np.pi)
    return solver.weights, states


# Return res x res matrix (representing true value function) running sample episodes to sample true value function
def get_true_value_func_samples(solver, res, pos_min, pos_max, vel_min, vel_max):
    env = solver.env
    true_samples = np.zeros((res, res))

    def get_return(pos, vel):
        state_traj = np.zeros(
            (
                solver.initial_state.size,
                int(solver.max_sim_time // env.control_timestep) + 1,
            )
        )
        state_traj[:, 0] = np.array([pos, vel])
        rewards = np.zeros(int(solver.max_sim_time // env.control_timestep))
        for t in range(int(solver.max_sim_time // env.control_timestep)):
            next_action = solver.get_epsilon_greedy_action(state_traj[:, t], epsilon=0)
            state_traj[:, t + 1] = env.get_next_state(next_action, state_traj[:, t])
            rewards[t] = env.get_reward(next_action, state_traj[:, t])
        state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)

        return_i = rewards[-1]
        for i in range(rewards.shape[0]):
            return_i *= solver.discount_factor
            return_i += rewards[rewards.shape[0] - i - 1]
        return return_i

    true_samples = sample(
        get_return, res=res, x_min=pos_min, x_max=pos_max, y_min=vel_min, y_max=vel_max
    )

    return true_samples


# ! rewrite
# Normalize both approximation and true value function, and plot them and generate statistics (TBD)
def compare_value_funcs(approx_weights, true_value_samples, solver):
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


def create_plots(
    solver, approx_weights, true_value_samples, res, states, x_min, x_max, y_min, y_max
):
    fig, axs = plt.subplots(1, 2)

    initial_state = solver.initial_state

    env = InvertedPendulum(
        params=solver.env.params,
        initial_state=initial_state,
        sim_timestep=solver.env.sim_timestep,
        control_timestep=solver.env.control_timestep,
    )

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

    max_torque = solver.max_torque
    torque_res = 11
    torques = np.linspace(-1 * max_torque, max_torque, torque_res)

    def find_max_q_vals_actions(pos, vel):
        sample_qs_at_x_y = np.zeros(torque_res)
        for i in range(torque_res):
            sample_qs_at_x_y[i] = solver.action_value_approx(torques[i], (pos, vel))
        return np.max(sample_qs_at_x_y)

    max_q_vals = sample(
        func=find_max_q_vals_actions,
        res=res,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )

    img = axs[1].imshow(max_q_vals, origin="lower", extent=[x_min, x_max, y_min, y_max])
    axs[1].set_title("Approximated value function\n(Maximum Q-value for each state)")
    fig.colorbar(img, ax=axs[1])
    axs[1].set_aspect((x_max - x_min) / (y_max - y_min))
    axs[1].set_xlabel("pos")
    axs[1].set_ylabel("vel")

    state_traj, action_traj = sample_trajectory(solver=solver)
    num_timesteps = int(solver.max_sim_time // solver.env.sim_timestep)

    plt.savefig("out/true_approx_q_val.png")

    plt.figure(2)
    plt.scatter(
        state_traj[0, :],
        state_traj[1, :],
        c=range(num_timesteps + 1),
        cmap="cool",
    )
    plt.title("Sample trajectory")
    plt.colorbar(label="Timestep")
    plt.xlabel("pos")
    plt.ylabel("vel")
    plt.savefig("out/sample_traj_states.png")

    # 2d histogram of states visited during q-learning
    plt.figure(3)
    plt.hist2d(
        x=states[0, :],
        y=states[1, :],
        bins=40,
        range=[[x_min, x_max], [y_min, y_max]],
        cmap="Greens",
        norm=matplotlib.colors.LogNorm(),
    )
    plt.title("States visited during Q-Learning")
    plt.colorbar(label="Frequency")
    plt.xlabel("pos")
    plt.ylabel("vel")
    plt.savefig("out/visited_states_during_learning.png")

    # Plot rewards recieved during sample trajectory
    plt.figure(4)
    rewards = np.zeros(num_timesteps + 1)
    timesteps = np.linspace(0, 1, rewards.size)
    for i in range(rewards.size):
        rewards[i] = solver.env.get_reward(action_traj[i], state_traj[:, i])
    plt.scatter(timesteps, rewards)
    plt.title("Rewards during sample trajectory")
    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.savefig("out/sample_traj_rewards.png")

    # Get policy
    plt.figure(5)

    def policy(pos, vel):
        return solver.get_epsilon_greedy_action([pos, vel], 0)

    policy_matrix = sample(
        func=policy, res=100, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
    )
    plt.imshow(
        policy_matrix, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="bwr"
    )
    plt.title("Policy")
    plt.colorbar(label="Applied torque")
    plt.xlabel("pos")
    plt.ylabel("vel")
    plt.savefig("out/policy.png")

    # Get sample trajectory animation
    get_animation(states=state_traj, actions=action_traj)

    plt.show()


# Run Q learning and compare with ground truth value function
solver = initialize_env_and_solver(sim_timestep=0.005, control_timestep=0.01)
approx_weights, states = run_q_learning(solver, num_episodes=200)
true_value_samples = get_true_value_func_samples(
    solver=solver, res=10, pos_min=0, pos_max=2 * np.pi, vel_min=-4, vel_max=4
)
create_plots(
    solver,
    approx_weights,
    true_value_samples,
    res=10,
    states=states,
    x_min=0,
    x_max=2 * np.pi,
    y_min=-4,
    y_max=4,
)

ls_weights = least_squares_fit(solver,
                               10,
                               x_min=0,
                               x_max=2 * np.pi,
                               y_min=-4,
                               y_max=4,
                               t_min = -1*solver.max_torque,
                               t_max=solver.max_torque)

with open("ls_weights.pkl", "wb") as file:
    pickle.dump(ls_weights, file)
    print("least squares weights saved")

# Find least squares fit to action-feature vector
#get_plot_least_squares(
#    solver=solver, res=10, x_min=0, x_max=2 * np.pi, y_min=-4, y_max=4
#)
