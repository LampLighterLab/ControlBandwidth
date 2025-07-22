import scipy.integrate
import numpy as np

class InvertedPendulum:
    # Point mass attached to a massless rigid rod
    
    # params is a dict containing keys "mass", "length", "gravity"
    def __init__(self, params, initial_state, value_func, sim_timestep=0.01, control_timestep=0.1):
        self.params = params
        self.angular_pos = initial_state[0]                # 0 is pointing straight up, positive clockwise
        self.angular_vel = initial_state[1]                # Positive velocity is clockwise
        self.value_func = value_func                       # function: [pos, vel] -> scalar
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
        return self.value_func(s)
    
    # The action is the applied torque
    def get_action_reward(self, a, s):
        next_state = self.get_next_state(a, s)
        return self.value_func(next_state)