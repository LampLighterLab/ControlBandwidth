import matplotlib.pyplot as plt
import numpy as np
import environment
import time
from environment import Actions
from value_estimator import MonteCarlo, TDLambda
from control_method import QLearning
from inverted_pendulum import InvertedPendulum

def plot_values(solver_object, value_func):
    arr = np.zeros((solver_object.env.SIZE_Y, solver_object.env.SIZE_X))
    for i in range(solver_object.env.SIZE_Y):
        for j in range(solver_object.env.SIZE_X):
            arr[i,j] = value_func((j,i))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()

def test_monte_carlo():
    rewards = {
            (5,5):1
            }
    terminal_states = [
        (5,5)
        ]
    env = environment.Gridworld(rewards, terminal_states, size=(10,8))

    def up_policy(s):
        return Actions.UP

    mc = MonteCarlo(env.generate_random_action, (0,0), env, discount_factor=0.9)

    start = time.time_ns()
    for i in range(n := 100):
        mc.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(mc, mc.get_state_value)
    
def test_tdlambda():
    rewards = {
            (5,5):1
            }
    terminal_states = [
        (5,5)
        ]
    env = environment.Gridworld(rewards, terminal_states, size=(10,8), control_freq=5)

    def up_policy(s):
        return Actions.UP

    td = TDLambda(env.generate_random_action, (0,0), env, discount_factor=0.9, learning_rate=0.1, trace_decay=0.5)

    start = time.time_ns()
    for i in range(n := 100):
        td.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(td, td.get_state_value)
    
def test_q_learning():
    rewards = {
            (7,8):1
            }
    terminal_states = [
        (7,8)
        ]
    env = environment.Gridworld(rewards, terminal_states, size=(10,10), control_freq=1)
    q_learning_object = QLearning(env, initial_state=(0,0), epsilon=0.3, discount_factor=0.9, step_size=0.5)
    start = time.time_ns()
    for i in range(n := 1000):
        q_learning_object.run_episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")

    # Used AI to generate code for plot only
    actions = [Actions.UP, Actions.DOWN, Actions.LEFT, Actions.RIGHT]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Action-Value Function")

    for ax, action, name in zip(axs.flat, actions, action_names):
        arr = np.zeros((q_learning_object.env.SIZE_Y, q_learning_object.env.SIZE_X))
        for i in range(q_learning_object.env.SIZE_Y):
            for j in range(q_learning_object.env.SIZE_X):
                arr[i, j] = q_learning_object.get_action_value(action, (j, i))

        im = ax.imshow(arr, origin="lower")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("x-coordinate")
        ax.set_ylabel("y-coordinate")
        ax.set_title(f"Action: {name}")

    plt.tight_layout()
    
    q_learning_object.get_optimal_path()
    plt.show()

def test_pendulum():
    params = {
        "mass": 1,
        "length": 1,
        "gravity": 1
        }
    def value_func(vel, pos):
        return 0
    pendulum_solver = InvertedPendulum(
        params, initial_state=(0.1,0), value_func=value_func, sim_timestep=0.005, control_timestep=0.1
        )
    
    states = [(0.1, 0.5*np.pi)]
    curr_state = states[0]
    for i in range(n := 50):
        curr_state = pendulum_solver.get_next_state(0, curr_state)
        states.append(curr_state)
    states = np.array(states)
    states = states.T
    states[1] = np.mod(states[1], 2 * np.pi)
    
    timesteps = np.arange(n+1)
    plt.scatter(states[0], states[1], c=timesteps)
    plt.title("State space")
    plt.colorbar(label="Timestep")
    plt.xlabel("vel")
    plt.ylabel("pos")
    plt.show()

#test_tdlambda()
#test_monte_carlo()
#test_q_learning()
test_pendulum()