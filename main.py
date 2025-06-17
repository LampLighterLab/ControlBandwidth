import matplotlib.pyplot as plt
import numpy as np
import environment
import time
from environment import Actions, Gridworld, IllegalActionException, TerminalStateException
from value_estimator import MonteCarlo, TDLambda

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
    env = environment.Gridworld(rewards, terminal_states, 10, 8)

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

test_tdlambda()
#test_monte_carlo()