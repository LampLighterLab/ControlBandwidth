# Run Monte Carlo and TDLambda value estimation on the gridworld

import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Gridworld
from value_estimator import MonteCarloDiscrete, TDLambdaDiscrete, MonteCarloContinuous
from inverted_pendulum import InvertedPendulum
from random import Random


def run_monte_carlo_discrete():
    rewards = {(6, 6): 1}
    terminal_states = [(6, 6)]
    env = Gridworld(rewards, terminal_states, size=(10, 8), control_freq=2)
    initial_state = (0, 0)

    mc = MonteCarloDiscrete(
        env.generate_random_action, initial_state, env, discount_factor=0.9
    )

    start = time.time_ns()
    for i in range(n := 100):
        mc.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")

    # Plot value function
    arr = np.zeros((mc.env.SIZE_Y, mc.env.SIZE_X))
    for i in range(mc.env.SIZE_Y):
        for j in range(mc.env.SIZE_X):
            arr[i, j] = mc.get_state_value((j, i))  # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()


def run_monte_carlo_continuous():
    random_generator = np.random.default_rng(123)
    params = {"mass": 1, "length": 1, "gravity": 1}
    initial_state = np.array([0.1, 0])

    def policy(s):
        return 0
        # return 0.1 * random_generator.random() - 0.05

    env = InvertedPendulum(
        params=params,
        initial_state=initial_state,
        sim_timestep=0.005,
        control_timestep=0.1,
    )
    mc = MonteCarloContinuous(
        initial_state, env, discount_factor=0, policy=policy, max_timestep=1000
    )

    # Run episodes
    start = time.time_ns()
    for i in range(n := 1):
        mc.run_episode()
        # print(f"Weights: {mc.weights}")
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")

    # Generate sample trajectory
    curr_state = mc.initial_state
    states = [curr_state]
    rewards = list()
    t = 0
    print(
        f"Total energy of first state is {0.5 * curr_state[1] ** 2 + (np.cos(curr_state[0]) - 1)}"
    )
    while t < mc.max_timestep:
        next_action = mc.get_next_action(curr_state)
        rewards.append(env.get_action_reward(next_action, curr_state))
        curr_state = env.get_next_state(next_action, curr_state)
        states.append(curr_state)
        t += env.control_timestep
    states = np.array(states)
    print(
        f"Total energy of last state is {0.5 * states[-1][1] ** 2 + (np.cos(states[-1][0]) - 1)}"
    )
    states = states.T
    states[0] = np.mod(states[0], 2 * np.pi)

    i = len(rewards)
    return_i = 0
    returns = list()
    while i > 0:
        i -= 1
        return_i = rewards[i] + mc.discount_factor * return_i
        returns.insert(0, return_i)

    # Plot value function
    x_min = 0
    x_max = 2 * np.pi
    y_min = -3
    y_max = 3
    X = np.linspace(x_min, x_max, 200)  # State-space coordinates
    Y = np.linspace(y_min, y_max, 200)
    Z = np.zeros((200, 200))
    a = 0  # Array coordinates
    b = 0
    for x in X:
        for y in Y:
            Z[b, a] = mc.state_value_approx((x, y))
            b += 1
        b = 0
        a += 1

    img = plt.imshow(
        Z, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto"
    )
    plt.colorbar(img)
    plt.xlabel("pos")
    plt.ylabel("vel")
    plt.suptitle(
        "Approximated value function,\nwith sample trajectory colored by returns"
    )

    # Plot sample trajectory
    plt.scatter(
        states[0][:-1], states[1][:-1], c=returns, edgecolor="white"
    )  # last state has no associated reward
    plt.show()


def run_tdlambda_discrete():
    rewards = {(6, 6): 1}
    terminal_states = [(6, 6)]
    env = Gridworld(rewards, terminal_states, size=(10, 8), control_freq=2)
    initial_state = (0, 0)

    td = TDLambdaDiscrete(
        env.generate_random_action,
        initial_state,
        env,
        discount_factor=0.9,
        learning_rate=0.1,
        trace_decay=0.5,
    )

    start = time.time_ns()
    for i in range(n := 100):
        td.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start) / n) / 1e6:.3f} ms per episode")

    # Plot value function
    arr = np.zeros((td.env.SIZE_Y, td.env.SIZE_X))
    for i in range(td.env.SIZE_Y):
        for j in range(td.env.SIZE_X):
            arr[i, j] = td.get_state_value((j, i))

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()


run_monte_carlo_continuous()
