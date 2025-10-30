# Run QLearning and REINFORCE control methods on the gridworld

import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Actions, Gridworld
from control_method import QLearning, ReinforceSoftmax, QLearningPendulum
from inverted_pendulum import InvertedPendulum


def run_q_learning():
    rewards = {
        (8, 8): 1,
    }
    terminal_states = [(8, 8)]
    env = Gridworld(rewards, terminal_states, size=(10, 10), control_freq=2)
    q_learning_object = QLearning(
        env, initial_state=(0, 0), epsilon=0.3, discount_factor=0.9
    )
    start = time.time_ns()
    for i in range(n := 100):
        q_learning_object.run_episode()
        print(i)
        print(q_learning_object.weights)
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")

    # Generate action-value plots
    actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Action-Value Function")

    im = None
    for ax, action, name in zip(axs.flat, actions, action_names):
        arr = np.zeros((q_learning_object.env.SIZE_Y, q_learning_object.env.SIZE_X))
        for i in range(q_learning_object.env.SIZE_Y):
            for j in range(q_learning_object.env.SIZE_X):
                arr[i, j] = q_learning_object.action_value_approx(action, (j, i))

        im = ax.imshow(arr, origin="lower", vmin=0, vmax=4)
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_title(f"Action: {name}")

    fig.colorbar(im, ax=axs[0, 0])
    plt.tight_layout()

    q_learning_object.get_optimal_path()
    plt.show()


def run_q_learning_pendulum():
    params = {"mass": 1, "length": 1, "gravity": 1, "damping": 0.1}
    initial_state = np.array([np.pi, 0.1])
    env = InvertedPendulum(
        params=params,
        initial_state=initial_state,
        sim_timestep=0.01,
        control_timestep=0.1,
    )
    q_solver = QLearningPendulum(
        env,
        initial_state=initial_state,
        epsilon=0.1,
        discount_factor=0.9,
        learning_rate=3e-5,
        max_sim_time=60,
    )

    fig, axs = plt.subplots(1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(16)

    # Run episodes and update Q approximation
    start = time.time_ns()
    for i in range(n := 200):
        q_solver.run_episode()
        if (i % 100) == 0:
            print(f"Running episode {i}")
    print(f"Final weights:\n{q_solver.weights}")
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")

    # Left, middle plots: Maximum Q value for each state, optimal action for each state

    x_min = -1 * np.pi
    x_max = np.pi
    y_min = -0.5
    y_max = 0.5
    res = 30  # Resolution of action-state space to sample
    max_torque = env.params["mass"] * env.params["gravity"] * env.params["length"]
    X = np.linspace(x_min, x_max, res)  # State-action-space coordinates
    Y = np.linspace(y_min, y_max, res)
    max_q_vals = np.zeros((res, res))
    max_q_actions = np.zeros((res, res))
    T = np.linspace(-1 * max_torque, max_torque, res)
    a = 0  # Array coordinates
    b = 0
    for x in X:
        for y in Y:
            max_q_val = q_solver.action_value_approx(T[0], (a, b))
            max_q_action = T[0]
            for t in T:
                q_val = q_solver.action_value_approx(t, (a, b))
                if q_val > max_q_val:
                    max_q_val = q_val
                    max_q_action = t
            max_q_vals[b, a] = max_q_val
            max_q_actions[b, a] = max_q_action
            b += 1
        b = 0
        a += 1

    img = axs[0].imshow(max_q_vals, origin="lower", extent=[x_min, x_max, y_min, y_max])
    axs[0].set_title("Maximum Q-value over all actions")
    fig.colorbar(img, ax=axs[0])
    axs[0].set_aspect((x_max - x_min) / (y_max - y_min))

    img = axs[1].imshow(
        max_q_actions, origin="lower", extent=[x_min, x_max, y_min, y_max]
    )
    axs[1].set_title("Action with the maximum Q-value")
    fig.colorbar(img, ax=axs[1])
    axs[1].set_aspect((x_max - x_min) / (y_max - y_min))

    # Right plot: sample episode

    # Generate sample episode using greedy policy
    original_epsilon = q_solver.epsilon
    q_solver.epsilon = 0
    curr_state = q_solver.initial_state
    states = [curr_state]
    rewards = list()
    t = 0
    while t < q_solver.max_sim_time:
        next_action = q_solver.get_epsilon_greedy_action(curr_state)
        rewards.append(env.get_action_reward(next_action, curr_state))
        curr_state = env.get_next_state(next_action, curr_state)
        states.append(curr_state)
        t += env.control_timestep
    states = np.array(states)
    states = states.T
    q_solver.epsilon = original_epsilon

    # Format positions to the domain [-pi, pi]
    def format_pos(pos):
        pos = np.mod(pos, 2 * np.pi)
        if pos > np.pi:
            return pos - (2 * np.pi)
        return pos

    format_pos_vectorized = np.vectorize(format_pos)
    states[0] = format_pos_vectorized(states[0])

    # Plot sample trajectory
    timesteps = np.linspace(0, q_solver.max_sim_time, len(states[0]) - 1)
    img = axs[2].scatter(
        states[0][:-1], states[1][:-1], c=timesteps, cmap="inferno"
    )  # last state has no associated reward
    axs[2].set_aspect(1)
    fig.colorbar(img, ax=axs[2])
    axs[2].set_xlabel("pos")
    axs[2].set_ylabel("vel")
    axs[2].set_title("Sample trajectory colored by timestep")

    for ax in axs:
        ax.set_xlabel("pos")
        ax.set_ylabel("vel")

    plt.show()


def run_reinforce_softmax():
    rewards = {(8, 8): 1}
    terminal_states = [(8, 8)]
    env = Gridworld(rewards, terminal_states, size=(10, 10), control_freq=2)

    # control_freq = 1: temperature must be >= 20
    # control_freq = 2: temperature must be >= 100 for good results (not get stuck between 2 states)
    # I don't know why this occurs
    reinforce_solver = ReinforceSoftmax(
        env, initial_state=(0, 0), discount_factor=0.9, step_size=0.2, temperature=100
    )
    start = time.time_ns()
    for i in range(n := 100):
        reinforce_solver.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")
    print("Sample path:")
    reinforce_solver.get_path()

    # Plot probabilities of picking an action for each state
    probs = np.zeros(shape=(4, env.SIZE_Y, env.SIZE_X))
    for x in range(env.SIZE_X):
        for y in range(env.SIZE_Y):
            action_probs_tuple = reinforce_solver.get_action_probs((x, y))
            valid_actions = action_probs_tuple[0]
            action_probs = action_probs_tuple[1]
            for i in range(len(valid_actions)):
                probs[valid_actions[i].value][y][x] = action_probs[i]

    action_names = ["Up", "Right", "Down", "Left"]
    fig, axs = plt.subplots(2, 2)
    for ax, action_name, i in zip(axs.flat, action_names, range(4)):
        ax.set_title(action_name)
        im = ax.imshow(probs[i], vmin=0, vmax=1, origin="lower")

    fig.suptitle("Action Probabilities by State")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.subplots_adjust(hspace=0.3)
    plt.show()


run_q_learning_pendulum()
