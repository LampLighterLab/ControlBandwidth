import scipy.integrate
import numpy as np

class InvertedPendulum:
    # Point mass attached to a massless rigid rod
    
    # params is a dict containing keys "mass", "length", "gravity"
    def __init__(self, params, initial_state, value_func, sim_timestep=0.01, control_timestep=0.1):
        self.params = params
        self.angular_vel = initial_state[0]                # Positive velocity is clockwise
        self.angular_pos = initial_state[1]                # 0 is pointing straight up, positive clockwise
        self.value_func = value_func                       # function: (vel, pos) -> scalar
        self.sim_timestep = sim_timestep
        self.control_timestep = control_timestep

    # z = (angular velocity, angular position)
    # T_app is applied torque on the pendulum
    def pendulum_dynamics(self, t, z, T_app):
        vel, pos = z
        m = self.params["mass"]
        l = self.params["length"]
        g = self.params["gravity"]
        I = m*l*l
        return [(g*np.sin(pos))/l + T_app/I, vel]
    
    # For testing
    def get_sol(self, t0, tf, z, T_app):
        sol = scipy.integrate.solve_ivp(self.pendulum_dynamics, [t0, tf], z, args=(T_app,), min_step=self.sim_timestep, max_step=self.sim_timestep)
        return sol
    
    def get_next_state(self, T_app, state):
        t = 0
        m = self.params["mass"]
        l = self.params["length"]
        g = self.params["gravity"]
        I = m*l*l
        while (t <= self.control_timestep - 0.5*self.sim_timestep):
            state = np.add(state, np.multiply(self.sim_timestep, [(g*np.sin(state[1]))/l + T_app/I, state[0]]))
            t += self.sim_timestep
        return state
    
    def get_state_reward(self, s):
        return self.value_func(s)
    
    # The action is the applied torque
    def get_action_reward(self, a, s):
        next_state = self.get_next_state(a, s)
        return self.value_func(next_state)