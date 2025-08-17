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
    def value_func(pos, vel):
        return 0
    initial_state = np.array([0.5*np.pi, 0.1])
    pendulum_solver = InvertedPendulum(
        params, initial_state=initial_state, value_func=value_func, sim_timestep=0.005, control_timestep=0.1
        )
    
    states = [initial_state]
    curr_state = states[0]
    for i in range(n := 50):
        curr_state = pendulum_solver.get_next_state(0, curr_state)
        states.append(curr_state)
    states = np.array(states)
    states = states.T
    states[0] = np.mod(states[0], 2 * np.pi)
    
    timesteps = np.arange(n+1)
    plt.scatter(states[0], states[1], c=timesteps)
    plt.title("State space")
    plt.colorbar(label="Timestep")
    plt.xlabel("pos")
    plt.ylabel("vel")
    plt.show()

#test_pendulum()

def test():
    X = np.linspace(-1*np.pi, np.pi, 100)
    Y = np.cos(X)
    
    fig, ax = plt.subplots()
    ax.plot(X, Y)
    plt.show()

test()