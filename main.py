import matplotlib.pyplot as plt
import numpy as np
import environment
import time
from environment import Actions, Gridworld, IllegalActionException, TerminalStateException
from value_estimator import MonteCarlo
from agent import Agent

def plot_values(value_func):
    arr = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            arr[j, i] = value_func((i,j))    # this is correct

    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.title("Value function")
    plt.show()

def test_gridworld_straight_line():
    rewards = {
        (0,4):2, 
        (0,6):4,
        (0,7):3,
        (0,8):5,
        (0,9):10
        }
    terminal_states = list()
    terminal_states.append((0,8))
    env = environment.Gridworld(rewards, terminal_states)

    agent_rewards = list()
    step_number = list()
    state = (0,0)
    try:
        for i in range(100):
            agent_rewards.append(env.reward(Actions.UP, state))
            state = env.next_state(Actions.UP, state)
            step_number.append(i)
    except TerminalStateException:
        print("Terminal state!")
    except IllegalActionException:
        print("Out of bounds!")

    plt.plot(step_number, agent_rewards)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.show()

def test_monte_carlo():
    rewards = {
            (9,9):1
            }
    terminal_states = [
        (9,9)
        ]
    env = environment.Gridworld(rewards, terminal_states)

    my_agent = Agent(dict(), (0, 0))
    # for i in range(10):
    #    for j in range(10):
    #        my_agent.update_policy_dict_entry((i, j), Actions.UP)
    my_agent.update_policy(my_agent.generate_random_action)
    
    mc = MonteCarlo(my_agent, env, 0.9)
    start = time.time_ns()
    for i in range(n := 1000):
        mc.episode()
    end = time.time_ns()
    print(f"Elapsed: {(end - start) / 1e6:.3f} ms")
    print(f"Averaged {((end - start)/n) / 1e6:.3f} ms per episode")
    plot_values(my_agent.value)
    
test_monte_carlo()