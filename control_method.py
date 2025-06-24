from environment import Actions
import random

class QLearning:
    # Implements the Q Learning policy improvement algorithm with a linear value function approximator
    # Q(a,s) = 0 initially for all a,s  
    def __init__(self, env, initial_state, epsilon, discount_factor, step_size, max_timestep=10000):
        self.env = env
        self.initial_state = initial_state
        self.epsilon = epsilon               # Random exploration factor in epsilon-greedy behavior policy
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.max_timestep = max_timestep
        self.state = initial_state
        self.action_value_dict = dict()      # map: (action, (x,y)) -> float
    
    def action_value(self, action, state):
        if ((action, state) in self.action_value_dict):
            return self.action_value_dict[(action, state)]
        return 0
    
    def episode(self):
        self.state = self.initial_state
        stepnum = 0
        straight_steps_remaining = 0
        next_action = Actions.UP            # declared here to keep `next_action` in scope. Initial action is arbitrary
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            
            # Generate next action, state according to epsilon-greedy policy
            if (straight_steps_remaining == 0):
                straight_steps_left = self.env.control_freq
                max_val = self.action_value(next_action, self.state)
                max_action = next_action
                for action in Actions:
                    if (self.action_value(action, self.state) > max_val):
                        max_val = self.action_value(action, self.state)
                        max_action = action
                r = random.random()
                if (r < self.epsilon):
                    next_action = self.env.generate_random_action(self.state)
                else:
                    next_action = max_action
                next_state = self.env.next_state(next_action, self.state)
            
            # Update action-value function
            max_val = self.action_value(next_action, next_state)
            max_action = next_action
            for action in Actions:
                if (self.action_value(action, next_state) > max_val):
                    max_val = self.action_value(action, next_state)
                    max_action = action
            max_q = self.env.action_reward(max_action, next_state)
            if ((next_action, self.state) in self.action_value_dict):
                self.action_value_dict[(next_action, self.state)] += self.step_size * (self.env.action_reward(next_action, self.state) + self.discount_factor*max_q - self.action_value_dict[(next_action, self.state)])
            else:
                self.action_value_dict[(next_action, self.state)] = self.step_size * (self.env.action_reward(next_action, self.state) + self.discount_factor*max_q)
            
            self.state = next_state
            stepnum += 1
            straight_steps_left -= 1
        
        # Handle terminal state




class LinearApprox:

    def __init__(self):
        self.weights = [0] * 7

    # Returns the feature vector (a 7-tuple) for action a and state s
    # [x'x, y'y, x'y, xy', x, y, 1]
    def features(self, a, s):
        x0 = s[0]
        y0 = s[1]
        next_state = self.env.next_state(a, s)
        x1 = next_state[0]
        y1 = next_state[1]
        return (x1*x0, y1*y0, x1*y0, x0*y1, x0, y0, 1)

    def action_value(self, a, s):
        feature_vector = self.features(a, s)
        sum = 0
        for i in range(len(feature_vector)):
            sum += feature_vector[i]*self.weights[i]
        return sum