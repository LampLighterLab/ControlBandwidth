import matplotlib.pyplot as plt
import numpy as np
from inverted_pendulum import InvertedPendulum

# Testing the pendulum simulation
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