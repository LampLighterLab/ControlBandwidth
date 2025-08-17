import numpy as np

class InvertedPendulum:
    # Point mass attached to a massless rigid rod
    
    # params is a dict containing keys "mass", "length", "gravity"
    def __init__(self, params, initial_state, sim_timestep=0.01, control_timestep=0.1):
        self.params = params
        self.angular_pos = initial_state[0]                # 0 is pointing straight up, positive clockwise
        self.angular_vel = initial_state[1]                # Positive velocity is clockwise
        self.sim_timestep = sim_timestep
        self.control_timestep = control_timestep

    # `T_app` is the applied torque on the pendulum
    def get_next_state(self, T_app, state):
        t = 0
        m = self.params["mass"]
        l = self.params["length"]
        g = self.params["gravity"]
        I = m*l*l
        while (t <= self.control_timestep - 0.5*self.sim_timestep):
            state = np.add(state, np.multiply(self.sim_timestep, [state[1], (g*np.sin(state[0]))/l + T_app/I]))
            t += self.sim_timestep
        return state
    
    def get_state_reward(self, s):
        return (np.exp(np.cos(s[0])) - abs(s[1]))
    
    # The action is the applied torque
    def get_action_reward(self, a, s):
        next_state = self.get_next_state(a, s)
        return self.get_state_reward(next_state)
    
    def feature_vector(self, s):
        pos = s[0]
        vel = s[1]
        KE = 0.5 * self.params["mass"] * vel**2
        PE = self.params["mass"] * self.params["gravity"] * self.params["length"] * (np.cos(pos) - 1)
        return np.array([np.cos(pos), np.sin(pos), vel, KE, PE, 1])
    
    def action_feature_vector(self, a, s):
        pos = s[0]
        vel = s[1]
        KE = 0.5 * self.params["mass"] * vel**2
        PE = self.params["mass"] * self.params["gravity"] * self.params["length"] * (np.cos(pos) - 1)
        return np.array([np.cos(pos), np.sin(pos), vel, KE, PE, 1, a*np.cos(pos), a*np.sin(pos), a*vel, a*KE, a*PE, a])