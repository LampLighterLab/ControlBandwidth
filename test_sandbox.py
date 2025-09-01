import matplotlib.pyplot as plt
import numpy as np
from inverted_pendulum import InvertedPendulum


# Testing the pendulum simulation
def test_pendulum():
    params = {"mass": 1, "length": 1, "gravity": 1}

    def value_func(pos, vel):
        return 0

    # initial_state = np.array([0.5 * np.pi, 0.1])
    params = {"mass": 1, "length": 1, "gravity": 1}
    initial_state = np.array([np.pi, 0.01])
    sim_timestep = 0.0001
    control_timestep = 0.01
    env = InvertedPendulum(
        params=params,
        initial_state=initial_state,
        sim_timestep=sim_timestep,
        control_timestep=control_timestep,
    )

    sim_time = 10
    state_traj = np.zeros((initial_state.size, int(sim_time // sim_timestep) + 1))
    state_traj[:, 0] = initial_state
    for i in range(int(sim_time // sim_timestep)):
        state_traj[:, i + 1] = env.get_next_state(0, state_traj[:, i])
    state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)

    plt.figure(1)
    plt.scatter(
        state_traj[0, :], state_traj[1, :], c=range(int(sim_time // sim_timestep) + 1)
    )
    plt.title("State space")
    plt.colorbar(label="Timestep")
    plt.xlabel("pos")
    plt.ylabel("vel")
    # plt.show()

    plt.figure(2)
    time_traj = np.array(range(int(sim_time // sim_timestep) + 1)) * sim_timestep
    potential_energy = env.get_potential_energy(state_traj)
    kinetic_energy = env.get_kinetic_energy(state_traj)

    plt.plot(time_traj, potential_energy)
    plt.plot(time_traj, kinetic_energy)
    plt.xlabel("time [s]")
    plt.ylabel("energy [j]")
    plt.legend(["potential", "kinetic"])
    plt.figure(3)
    plt.plot(time_traj, potential_energy + kinetic_energy)
    plt.ylabel("totla energy [j]")
    plt.xlabel("time [s]")
    plt.show()


# test_pendulum()


def test():
    X = np.linspace(-1 * np.pi, np.pi, 100)
    Y = np.cos(X)

    fig, ax = plt.subplots()
    ax.plot(X, Y)
    plt.show()


test_pendulum()
