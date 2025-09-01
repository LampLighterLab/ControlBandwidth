import numpy as np


class InvertedPendulum:
    # Point mass attached to a massless rigid rod

    # params is a dict containing keys "mass", "length", "gravity"
    def __init__(self, params, initial_state, sim_timestep=0.01, control_timestep=0.1):
        self.params = params
        self.angular_pos = initial_state[0]
        # ! 0 is pointing straight down (passively stable equilibrium point), positive follows RHR!
        self.angular_vel = initial_state[1]
        self.sim_timestep = sim_timestep
        self.control_timestep = control_timestep

    # `T_app` is the applied torque on the pendulum
    def get_next_state(self, torque, state):
        mass = self.params["mass"]
        length = self.params["length"]
        gravity = self.params["gravity"]
        inertia = mass * length * length
        timestep = self.sim_timestep
        for _ in range(int(self.control_timestep // self.sim_timestep)):
            state += timestep * np.array(
                [state[1], -(gravity * np.sin(state[0])) / length + torque / inertia]
            )
        return state

    def get_potential_energy(self, state):
        PE = (
            self.params["mass"]
            * self.params["gravity"]
            * self.params["length"]
            * (1 - np.cos(state[0, :]))
        )
        return PE

    def get_kinetic_energy(self, state):
        KE = self.params["mass"] / 2 * (self.params["length"] * state[1, :]) ** 2
        return KE

    def get_energy(self, state):
        return self.get_potential_energy(state) + self.get_kinetic_energy(state)

    def get_state_reward(self, s):
        return np.exp(np.cos(s[0])) - 0.5 * abs(s[1])

    # The action is the applied torque
    def get_action_reward(self, a, s):
        next_state = self.get_next_state(a, s)
        return self.get_state_reward(next_state)

    def feature_vector(self, s):
        pos = s[0]
        vel = s[1]
        KE = 0.5 * self.params["mass"] * vel**2
        PE = (
            self.params["mass"]
            * self.params["gravity"]
            * self.params["length"]
            * (1 - np.cos(pos))
        )
        return np.array([np.cos(pos), np.sin(pos), vel, abs(vel), KE, PE, 1])

    def action_feature_vector(self, a, s):
        pos = s[0]
        vel = s[1]
        KE = 0.5 * self.params["mass"] * vel**2
        PE = (
            self.params["mass"]
            * self.params["gravity"]
            * self.params["length"]
            * (1 - np.cos(pos))
        )
        return np.array(
            [
                np.cos(pos),
                np.sin(pos),
                vel,
                abs(vel),
                KE,
                PE,
                1,
                a * np.cos(pos),
                a * np.sin(pos),
                a * vel,
                a * abs(vel),
                a * KE,
                a * PE,
                a,
            ]
        )
