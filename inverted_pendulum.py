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

    def compute_dynamics(self, state, torque):
        acceleration = (
            -(self.params["gravity"] * np.sin(state[0])) / self.params["length"]
            - self.params["damping"] * state[1]
            + torque / (self.params["mass"] * self.params["length"] ** 2)
        )
        return np.array([state[1], acceleration])

    def integrate_euler(self, state, torque, timestep):
        return state + timestep * self.compute_dynamics(state, torque)

    def integrate_rk4(self, state, torque, timestep):
        next_state = state
        k1 = self.compute_dynamics(next_state, torque)
        k2 = self.compute_dynamics(next_state + 0.5 * timestep * k1, torque)
        k3 = self.compute_dynamics(next_state + 0.5 * timestep * k2, torque)
        k4 = self.compute_dynamics(next_state + timestep * k3, torque)
        next_state += (timestep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return next_state

    def get_next_state(self, torque, state):
        for _ in range(int(self.control_timestep // self.sim_timestep)):
            state = self.integrate_rk4(state, torque, self.sim_timestep)
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

    def get_reward(self, a, s):
        return (1 - np.cos(s[0])) - 0.1 * (a**2) - 0.1 * (s[1] ** 2)

    # The action is the applied torque
    # ! stop using this because it computes next_state unnecessarily
    def get_action_reward(self, a, s):
        next_state = self.get_next_state(a, s)
        return self.get_reward(next_state)

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
        # Testing removing m,g,l from calculation because it's the same up to a constant
        pos = s[0]
        vel = s[1]
        # KE = 0.5 * vel**2
        # PE = 1 - np.cos(pos)
        return np.array(
            [
                1,
                np.cos(pos - np.pi),
                np.sin(pos - np.pi),
                vel,
                vel**2,
                a,
            ]
        )
