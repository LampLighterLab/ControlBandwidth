import matplotlib.pyplot as plt
import numpy as np
import time
from environment import Gridworld
from control_method import QLearning


def test_select_max():
    # Testing code used to select max torque from a range

    def func(x):
        return -1 * ((x - 0.5) ** 2)

    max_torque = 1
    sample_torques = np.linspace(-1 * max_torque, max_torque, 15)
    sample_q_vals = np.zeros(sample_torques.size)
    for i in range(sample_torques.size):
        sample_q_vals[i] = func(sample_torques[i])
    print(sample_torques[np.argmax(sample_q_vals)])


# test_select_max()
