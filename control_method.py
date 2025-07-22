from environment import Actions, Gridworld
import numpy as np
import math
import random

class QLearning:
    # Implements the Q Learning algorithm
    # Q(a,s) = 0 initially for all a,s  
    def __init__(self, env, initial_state, epsilon, discount_factor, step_size, max_timestep=1000):
        self.env = env
        self.initial_state = initial_state
        self.epsilon = epsilon                  # Random exploration factor in epsilon-greedy behavior policy
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.max_timestep = max_timestep
        self.state = initial_state
        self.action_value_dict = dict()         # map: (action, (x,y)) -> float
    
    def get_action_value(self, action, state):
        if ((action, state) in self.action_value_dict):
            return self.action_value_dict[(action, state)]
        return 0
    
    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        next_action = Actions.UP            # declared here to keep `next_action` in scope. Initial action is arbitrary
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            # Generate next action, state according to epsilon-greedy policy
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
            next_state = self.env.get_next_state(next_action, self.state, self.env.control_freq)
            
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
        
        # ! terminal state is not handled
    
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
            curr_state = self.env.get_next_state(max_action, curr_state, 1)
        if curr_state in path:
            print("Greedy path is cyclic! No path can be found with current Q(s,a)")
        if curr_state in self.env.terminal_states:
            path.append(curr_state)
            print("Path:")
            print(path)
            print(f"Path length is {len(path)}")


class Reinforce:
    # REINFORCE algorithm using the softmax policy
    
    def __init__(self, env, initial_state, discount_factor, step_size, temperature, max_timestep=1000):
        self.env = env
        self.initial_state = initial_state
        self.state = initial_state
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.temperature = temperature
        self.max_timestep = max_timestep
        
        self.weights = [0] * len(Gridworld.feature_vector(Actions.UP, initial_state))
        
        self.random_generator = np.random.default_rng(123456)

    # Return the set of valid actions and their probabilities according to policy
    def get_action_probs(self, state):
        valid_actions = self.env.get_valid_actions(state)
        action_probs = list()
        sum_weights = 0
        for action in valid_actions:
            feature_scalar = np.dot(self.weights, Gridworld.feature_vector(action, state))
            weight = math.exp(feature_scalar / self.temperature)
            action_probs.append(weight)
            sum_weights += weight
        for i in range(len(action_probs)):
            action_probs[i] = action_probs[i] / sum_weights
        return (valid_actions, action_probs)
    
    # Calculate value of score function to be used in updating weights
    def score_function(self, action, state):
        action_probs_tuple = self.get_action_probs(state)
        valid_actions = action_probs_tuple[0]
        action_probs = action_probs_tuple[1]
        gradient_log_policy = Gridworld.feature_vector(action, state)
        for i in range(len(valid_actions)):
            gradient_log_policy = np.subtract(gradient_log_policy,
                                              np.multiply(action_probs[i], Gridworld.feature_vector(valid_actions[i], state)))
        return gradient_log_policy
    
    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        states = list()
        actions = list()
        rewards = list()
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            # Generate next action, state according to parameterized policy
            action_probs_tuple = self.get_action_probs(self.state)
            valid_actions = action_probs_tuple[0]
            action_probs = action_probs_tuple[1]
            next_action = self.random_generator.choice(valid_actions, p=action_probs)
            states.append(self.state)
            actions.append(next_action)
            rewards.append(self.env.get_action_reward(next_action, self.state))
            self.state = self.env.get_next_state(next_action, self.state, self.env.control_freq)
            stepnum += 1
        
        # Handle terminal state
        if (self.state in self.env.terminal_states):
            states.append(self.state)
        
        # Calculate returns and update weights
        i = len(states) - 1
        return_i = 0
        while (i > 0):
            i -= 1
            return_i = rewards[i] + self.discount_factor*return_i
            self.weights = np.add(self.weights,
                                  self.step_size*return_i*self.score_function(actions[i], states[i]))
    
    # Generate a sample path without updating weights
    def get_path(self):
        states = list()
        stepnum = 0
        state = self.initial_state
        while (stepnum < self.max_timestep and state not in self.env.terminal_states):
            # Generate next action, state according to parameterized policy
            action_probs_tuple = self.get_action_probs(state)
            valid_actions = action_probs_tuple[0]
            action_probs = action_probs_tuple[1]
            next_action = self.random_generator.choice(valid_actions, p=action_probs)
            states.append(state)
            state = self.env.get_next_state(next_action, state, 1)
            stepnum += 1
        if (state in self.env.terminal_states):
            states.append(state)
        print(states)