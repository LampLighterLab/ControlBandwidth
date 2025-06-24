import matplotlib.pyplot as plt
import numpy as np
import environment
import time
from environment import Actions
from value_estimator import MonteCarlo, TDLambda
from control_method import QLearning

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
    env = environment.Gridworld(rewards, terminal_states, 10, 8)

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
    env = environment.Gridworld(rewards, terminal_states, 10, 8, 5)

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
            (5,5):1
            }
    terminal_states = [
        (5,5)
        ]
    env = environment.Gridworld(rewards, terminal_states, 10, 8)
    q = QLearning(env, (0,0), 0.1, 1, 0.5)
    start = time.time_ns()
    for i in range(n := 100):
        q.episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    
    solver_obj = q

    # Used AI to generate code for plot only
    actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
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
    plt.show()


#test_tdlambda()
#test_monte_carlo()
test_q_learning()