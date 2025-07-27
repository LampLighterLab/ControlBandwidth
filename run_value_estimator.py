# Run Monte Carlo and TDLambda value estimation on the gridworld

import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Gridworld
from value_estimator import MonteCarloDiscrete, TDLambdaDiscrete, MonteCarloContinuous
from inverted_pendulum import InvertedPendulum

def run_monte_carlo_discrete():
    rewards = {
            (6,6):1
            }
    terminal_states = [
        (6,6)
        ]
    env = Gridworld(rewards, terminal_states, size=(10,8), control_freq=2)
    initial_state = (0,0)

    mc = MonteCarloDiscrete(env.generate_random_action, initial_state, env, discount_factor=0.9)

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

def run_monte_carlo_continuous():
    params = {"mass":1, "length":1, "gravity":1}
    initial_state = np.array([0.01, 0])
    env = InvertedPendulum(params=params, initial_state=initial_state, sim_timestep = 0.01, control_timestep= 0.1)
    mc = MonteCarloContinuous(initial_state, env, discount_factor=0.9, max_timestep=20)
    
    start = time.time_ns()
    for i in range(n := 100):
        mc.run_episode()
        print(f"Weights: {mc.weights}")
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    
    # Plot value function
    x_min = -1 * np.pi
    x_max = np.pi
    y_min = -2
    y_max = 2
    X = np.linspace(x_min, x_max, 200) # State-space coordinates
    Y = np.linspace(y_min, y_max, 200)
    Z = np.zeros((200, 200))
    a = 0   # Array coordinates
    b = 0
    for x in X:
        for y in Y:
            Z[b, a] = mc.state_value_approx((x, y))
            b += 1
        b = 0
        a += 1

    plt.imshow(Z, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    plt.colorbar()
    plt.xlabel("pos")
    plt.ylabel("vel")
    plt.title("Value function")
    plt.show()

def run_tdlambda_discrete():
    rewards = {
            (6,6):1
            }
    terminal_states = [
        (6,6)
        ]
    env = Gridworld(rewards, terminal_states, size=(10,8), control_freq=2)
    initial_state = (0,0)

    td = TDLambdaDiscrete(env.generate_random_action, initial_state, env, discount_factor=0.9, learning_rate=0.1, trace_decay=0.5)

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

run_monte_carlo_continuous()