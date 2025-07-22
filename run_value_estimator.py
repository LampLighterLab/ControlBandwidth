# Run Monte Carlo and TDLambda value estimation on the gridworld

import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Gridworld
from value_estimator import MonteCarlo, TDLambda

def run_monte_carlo():
    rewards = {
            (6,6):1
            }
    terminal_states = [
        (6,6)
        ]
    env = Gridworld(rewards, terminal_states, size=(10,8), control_freq=2)
    initial_state = (0,0)

    mc = MonteCarlo(env.generate_random_action, initial_state, env, discount_factor=0.9)

    start = time.time_ns()
    for i in range(n := 100):
        mc.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    
    # Plot value function
    arr = np.zeros((mc.env.SIZE_Y, mc.env.SIZE_X))
    for i in range(mc.env.SIZE_Y):
        for j in range(mc.env.SIZE_X):
            arr[i,j] = mc.get_state_value((j,i))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()

def run_tdlambda():
    rewards = {
            (6,6):1
            }
    terminal_states = [
        (6,6)
        ]
    env = Gridworld(rewards, terminal_states, size=(10,8), control_freq=2)
    initial_state = (0,0)

    td = TDLambda(env.generate_random_action, initial_state, env, discount_factor=0.9, learning_rate=0.1, trace_decay=0.5)

    start = time.time_ns()
    for i in range(n := 100):
        td.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")

    # Plot value function
    arr = np.zeros((td.env.SIZE_Y, td.env.SIZE_X))
    for i in range(td.env.SIZE_Y):
        for j in range(td.env.SIZE_X):
            arr[i,j] = td.get_state_value((j,i))

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()