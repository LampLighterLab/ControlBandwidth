import matplotlib.pyplot as plt
import numpy as np
import environment
from environment import Actions, IllegalActionException, TerminalStateException


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

test_gridworld_straight_line()