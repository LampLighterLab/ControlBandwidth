from environment import Actions
import random

class QLearning:
    # Implements the Q Learning policy improvement algorithm
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
    
    def get_action_value(self, action, state):
        if ((action, state) in self.action_value_dict):
            return self.action_value_dict[(action, state)]
        return 0
    
    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        straight_steps_remaining = 0
        next_action = Actions.UP            # declared here to keep `next_action` in scope. Initial action is arbitrary
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            # Generate next action, state according to epsilon-greedy policy
            if (straight_steps_remaining == 0):
                straight_steps_left = self.env.control_freq
                max_q_val = self.get_action_value(next_action, self.state)
                max_action = next_action
                for action in self.env.get_valid_actions(self.state):
                    if (self.get_action_value(action, self.state) > max_q_val):
                        max_q_val = self.get_action_value(action, self.state)
                        max_action = action
                r = random.random()
                if (r < self.epsilon):
                    next_action = self.env.generate_random_action(self.state)
                else:
                    next_action = max_action
            next_state = self.env.get_next_state(next_action, self.state)
            
            # Update action-value function
            max_q_val = self.get_action_value(next_action, next_state)
            max_action = next_action
            for action in self.env.get_valid_actions(next_state):
                if (self.get_action_value(action, next_state) > max_q_val):
                    max_q_val = self.get_action_value(action, next_state)
                    max_action = action
            if ((next_action, self.state) in self.action_value_dict):
                self.action_value_dict[(next_action, self.state)] += self.step_size * (self.env.get_action_reward(next_action, self.state) + self.discount_factor*max_q_val - self.action_value_dict[(next_action, self.state)])
            else:
                self.action_value_dict[(next_action, self.state)] = self.step_size * (self.env.get_action_reward(next_action, self.state) + self.discount_factor*max_q_val)
            
            self.state = next_state
            stepnum += 1
            straight_steps_left -= 1
        
        # Handle terminal state
    
    def get_optimal_path(self):
        path = list()
        curr_state = self.initial_state
        while (curr_state not in path and curr_state not in self.env.terminal_states):
            path.append(curr_state)
            max_action = self.env.get_valid_actions(curr_state)[0]
            max_q_val = self.get_action_value(max_action, curr_state)
            for action in self.env.get_valid_actions(curr_state):
                if (self.get_action_value(action, curr_state) > max_q_val):
                    max_q_val = self.get_action_value(action, curr_state)
                    max_action = action
            curr_state = self.env.get_next_state(max_action, curr_state)
        if curr_state in path:
            print("Greedy path is cyclic! No path can be found with current Q(s,a)")
        if curr_state in self.env.terminal_states:
            path.append(curr_state)
            print("Path:")
            print(path)
            print(f"Path length is {len(path)}")
