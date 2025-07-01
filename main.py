import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Actions, Gridworld
from value_estimator import MonteCarlo, TDLambda
from control_method import QLearning
import math

def plot_values(solver_obj):
    arr = np.zeros((solver_obj.env.SIZE_Y, solver_obj.env.SIZE_X))
    for i in range(solver_obj.env.SIZE_Y):
        for j in range(solver_obj.env.SIZE_X):
            arr[i,j] = solver_obj.value((j,i))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()

def test_monte_carlo():
    rewards = {
            (0,5):1
            }
    terminal_states = [
        (0,5)
        ]
    env = Gridworld(rewards, terminal_states, 10, 8)

    def up_policy(s):
        return Actions.UP

    mc = MonteCarlo(up_policy, (0,0), env, 0.9, 1000)

    start = time.time_ns()
    for i in range(n := 1):
        mc.episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(mc)
    
def test_tdlambda():
    rewards = {
            (5,5):1
            }
    terminal_states = [
        (5,5)
        ]
    env = Gridworld(rewards, terminal_states, 10, 8, 5)

    def up_policy(s):
        return Actions.UP

    td = TDLambda(env.generate_random_action, (0,0), env, 0.9, 0.1, 0.5, 1000)

    start = time.time_ns()
    for i in range(n := 100):
        td.episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(td)
    
def test_q_learning():
    rewards = {
            (7,8):1
            }
    terminal_states = [
        (7,8)
        ]
    env = Gridworld(rewards, terminal_states, 10, 10)
    def obs_func(s):
            x = s[0]-7
            y = s[1]-8
            return [x**2, y**2, 1]
    q = QLearning(env, (0,0), 0.5, 0.3, 0.5, obs_func)
    start = time.time_ns()
    for i in range(n := 500):
        q.episode()
    print(q.weights)
    q.optimal_path()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    
    solver_obj = q
    arr = np.zeros((solver_obj.env.SIZE_Y, solver_obj.env.SIZE_X))
    for i in range(solver_obj.env.SIZE_Y):
        for j in range(solver_obj.env.SIZE_X):
            arr[i,j] = solver_obj.state_value_approx((j,i))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()
    

    # Used AI to generate code for plot only
"""     actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Action-Value Function")

    for ax, action, name in zip(axs.flat, actions, action_names):
        arr = np.zeros((solver_obj.env.SIZE_Y, solver_obj.env.SIZE_X))
        for i in range(solver_obj.env.SIZE_Y):
            for j in range(solver_obj.env.SIZE_X):
                arr[i, j] = solver_obj.action_value(action, (j, i))

        im = ax.imshow(arr, origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_title(f"Action: {name}")

    plt.tight_layout()
    
    q.optimal_path()
    plt.show() """


#test_tdlambda()
#test_monte_carlo()
#test_q_learning()

def test_q_hyperparams():
    rewards = {
            (7,8):1
            }
    terminal_states = [
        (7,8)
        ]
    env = Gridworld(rewards, terminal_states, 10, 10)
    def obs_func(s):
            x = s[0]-7
            y = s[1]-8
            return [x**2, y**2, 1]
    def symlog(x):
        if (x>1):
            return math.log10(x)
        elif (x>-1):
            return 0
        else:
            return -1 * math.log10(-1 * x)

    import statistics
    hyperparam_space = np.zeros((10, 11))
    e = 1
    d = 0
    while (e <= 9):
        while (d <= 10):
            start = time.time_ns()
            for trials in range(5):
                q_solver = QLearning(env, (0,0), e/10, d/10, 0.5, obs_func)
                ratio = list()
                for episodes in range(500):
                    q_solver.episode()
                ratio.append(q_solver.state_value_approx((7,8)) / q_solver.state_value_approx((0,0)))
            hyperparam_space[e][d] = statistics.median(ratio)
            d += 1
            end = time.time_ns()
            print(d)
            print(e)
            print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
        e += 1
        d = 0
    
    print(hyperparam_space)
    
    x = np.linspace(0.0, 1.0, num=11)
    y = np.linspace(0.1, 1.0, num=10)

    extent = [x[0], x[-1], y[0], y[-1]]

    plt.imshow(
        hyperparam_space,
        extent=extent,
        aspect="auto",
        origin="lower"
    )
    
    plt.imshow(hyperparam_space, origin="lower")
    plt.colorbar()
    plt.xlabel("discount factor")
    plt.ylabel("epsilon")
    plt.title("Ratio of value func approx of the terminal state to initial state")
    plt.show()

test_q_hyperparams()