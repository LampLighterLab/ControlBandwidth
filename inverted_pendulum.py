import numpy as np


class InvertedPendulum:
    # Point mass attached to a massless rigid rod

    # params is a dict containing keys "mass", "length", "gravity"
    def __init__(self, params, initial_state, sim_timestep=0.01, control_timestep=0.1):
        if "damping" not in params:
            params = {**params, "damping": 0.0}
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
        state = state.copy()
        for _ in range(int(self.control_timestep // self.sim_timestep)):
            state = self.integrate_rk4(state, torque, self.sim_timestep)
        return state

    def get_potential_energy(self, state):
        PE = (
            self.params["mass"]
            * self.params["gravity"]
            * self.params["length"]
            * (1 - np.cos(state[0]))
        )
        return PE

    def get_kinetic_energy(self, state):
        KE = self.params["mass"] / 2 * (self.params["length"] * state[1]) ** 2
        return KE

    def get_energy(self, state):
        return self.get_potential_energy(state) + self.get_kinetic_energy(state)

    def get_reward(self, a, s):
        E = self.get_energy(s)
        E_ideal = self.get_energy([np.pi, 0])
        return (
            (1 - np.cos(s[0]))
            - 0.1 * (a**2)
            - 0.1 * (s[1] ** 2)
            - 0.2 * abs(E - E_ideal)
        )

    # The action is the applied torque
    # ! stop using this because it computes next_state unnecessarily
    def get_action_reward(self, a, s):
        next_state = self.get_next_state(a, s)
        return self.get_reward(next_state)

    def state_feature_vector(self, s):
        pos = s[0]
        vel = s[1]

        theta_tilde = np.arctan2(np.sin(pos - np.pi), np.cos(pos - np.pi))

        KE = 0.5 * self.params["mass"] * vel**2
        PE = (
            self.params["mass"]
            * self.params["gravity"]
            * self.params["length"]
            * (1 - np.cos(pos))
        )
        E = KE + PE
        E_ideal = self.get_energy([np.pi, 0])
        energy_delta = (E - E_ideal) / E_ideal

        return np.array(
            [
                np.sin(theta_tilde),
                np.cos(theta_tilde),
                theta_tilde,
                theta_tilde**2,
                vel,
                vel**2,
                energy_delta,
                np.sign(energy_delta),
                np.sign(0.5*E_ideal - E),
                #np.sign(theta_tilde * vel),
                np.sign(energy_delta * theta_tilde * vel),
            ]
        )

    def action_feature_vector(self, a, s):
        # Testing removing m,g,l from calculation because it's the same up to a constant
        pos = s[0]
        vel = s[1]
        E = self.get_energy(s)
        E_ideal = self.get_energy([np.pi, 0])
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
                a**2,
                E - E_ideal,
                a * (E - E_ideal) ** 2,
            ]
        )

    # [
    #                 1,
    #                 np.cos(pos - np.pi),
    #                 np.sin(pos - np.pi),
    #                 vel,
    #                 vel**2,
    #                 a,
    #                 a**2,
    #                 a * np.cos(pos - np.pi),
    #                 a * np.sin(pos - np.pi),
    #                 E - E_ideal,
    #                 a * (E - E_ideal) ** 2,
    #                 (E - E_ideal) ** 2,
    #             ]

    def linearize(self, state, torque):
        cos_theta = np.cos(state[0])

        A = np.array(
            [
                [0.0, 1.0],
                [
                    -(self.params["gravity"] / self.params["length"]) * cos_theta,
                    -self.params["damping"],
                ],
            ]
        )
        B = np.array(
            [[0.0], [1.0 / (self.params["mass"] * self.params["length"] ** 2)]]
        )
        return A, B
