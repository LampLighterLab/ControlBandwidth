from environment import Actions
import numpy as np
from numpy.linalg import LinAlgError
import math

class QLearning:
    # Implements the Q Learning policy improvement algorithm with state-value function approximation

    def __init__(self, env, obs_func, initial_state, epsilon, discount_factor, max_timestep=10000):
        self.env = env
        self.initial_state = initial_state
        self.epsilon = epsilon               # Random exploration factor in epsilon-greedy behavior policy
        self.discount_factor = discount_factor
        self.max_timestep = max_timestep
        self.state = initial_state
        self.action_value_dict = dict()      # map: (action, (x,y)) -> float
        self.random_generator = np.random.default_rng(1234526)
        
        self.obs_vector = obs_func           # function: tuple (state) -> tuple (feature vector)
        self.weights = [0] * len(obs_func(initial_state))
    
    def state_value_approx(self, s):
        sum = 0
        obs_s = self.obs_vector(s)
        for i in range(len(self.weights)):
            sum += self.weights[i] * obs_s[i]
        return sum
    
    def action_value_approx(self, a, s):
        next_state = self.env.get_next_state(a, s)
        return self.env.get_state_reward(s) + self.discount_factor*self.env.get_state_reward(next_state)
    
    # Find least squares solution for weights using TD(0) error
    # states, rewards are lists of their values in an episode
    def update_weights(self, states, rewards):
        outer_prod = np.zeros((len(self.weights), len(self.weights)))
        r_vec = np.zeros(len(self.weights))
        for i in range(len(states)-1):
            a = np.array(self.obs_vector(states[i]))
            b = np.array(self.obs_vector(states[i]) - np.multiply(self.discount_factor, self.obs_vector(states[i+1])))
            a = np.reshape(a, (a.shape[0], 1))
            b = np.reshape(b, (1, b.shape[0]))
            outer_prod = np.add(outer_prod, a * b)
        for i in range(len(states)):
            r_vec = np.add(r_vec, np.multiply(rewards[i], self.obs_vector(states[i])))
        # "cheating" to get around non-invertibility
        try:
            outer_prod = np.linalg.inv(outer_prod)
        except LinAlgError:
            outer_prod = np.add(outer_prod, np.multiply(0.0001, np.identity(len(self.weights))))
            outer_prod = np.linalg.inv(outer_prod)
        self.weights = np.matmul(outer_prod, r_vec)
    
    def get_action_value(self, action, state):
        if ((action, state) in self.action_value_dict):
            return self.action_value_dict[(action, state)]
        return 0
    
    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        straight_steps_remaining = 0
        next_action = Actions.UP            # declared here to keep `next_action` in scope. Initial action is arbitrary
        states = list()
        rewards = list()
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            # Generate next action, state according to epsilon-greedy policy
            if (straight_steps_remaining == 0):
                straight_steps_remaining = self.env.control_freq
                max_q_val = self.action_value_approx(next_action, self.state)
                max_action = next_action
                for action in self.env.get_valid_actions(self.state):
                    if (self.action_value_approx(action, self.state) > max_q_val):
                        max_q_val = self.action_value_approx(action, self.state)
                        max_action = action
                r = self.random_generator.random()
                if (r < self.epsilon):
                    next_action = self.env.generate_random_action(self.state)
                else:
                    next_action = max_action
                states.append(self.state)
                rewards.append(self.env.get_action_reward(next_action, self.state))
            next_state = self.env.get_next_state(next_action, self.state)
            
            self.state = next_state
            stepnum += 1
            straight_steps_remaining -= 1
        self.update_weights(states, rewards)
        
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


class Reinforce:
    # REINFORCE algorithm using the softmax policy
    
    def __init__(self, env, initial_state, discount_factor, step_size, temperature, max_timestep=10000):
        self.env = env
        self.initial_state = initial_state
        self.state = initial_state
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.temperature = temperature
        self.max_timestep = max_timestep
        
        def feature_vector(a, s):
            x = s[0] - 7
            y = s[1] - 8
            i = a.value
            num_state_features = 6
            return [0]*num_state_features*(i) + [x**2, y**2, 1, x, y, x*y] + [0]*num_state_features*(3-i)
        self.feature_vector = feature_vector
        self.weights = [0] * len(self.feature_vector(Actions.UP, initial_state))
        
        self.random_generator = np.random.default_rng(123456)

    # Return the set of valid actions and their probabilities according to policy
    def get_action_probs(self, state):
        valid_actions = self.env.get_valid_actions(state)
        action_probs = list()
        sum_weights = 0
        for action in valid_actions:
            feature_scalar = np.dot(self.weights, self.feature_vector(action, state))
            weight = math.exp(feature_scalar / self.temperature)
            action_probs.append(weight)
            sum_weights += weight
        for i in range(len(action_probs)):
            action_probs[i] = action_probs[i] / sum_weights
        return (valid_actions, action_probs)
    
    # Calculate value of score function to be used in updating weights. 
    def score_function(self, action, state):
        action_probs_tuple = self.get_action_probs(state)
        valid_actions = action_probs_tuple[0]
        action_probs = action_probs_tuple[1]
        gradient_log_policy = self.feature_vector(action, state)
        for i in range(len(valid_actions)):
            gradient_log_policy = np.subtract(gradient_log_policy,
                                              np.multiply(action_probs[i], self.feature_vector(valid_actions[i], state)))
        return gradient_log_policy
    
    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        straight_steps_remaining = 0
        states = list()
        actions = list()
        rewards = list()
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            # Generate next action, state according to parameterized policy
            if (straight_steps_remaining == 0):
                straight_steps_remaining = self.env.control_freq
                action_probs_tuple = self.get_action_probs(self.state)
                valid_actions = action_probs_tuple[0]
                action_probs = action_probs_tuple[1]
                next_action = self.random_generator.choice(valid_actions, p=action_probs)
                states.append(self.state)
                actions.append(next_action)
                rewards.append(self.env.get_action_reward(next_action, self.state))
                #print(self.state)
                #print(valid_actions)
                #print(action_probs)
                #print(next_action)
                #print()
            self.state = self.env.get_next_state(next_action, self.state)
            stepnum += 1
            straight_steps_remaining -= 1
        
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
    
    def get_path(self):
        states = list()
        stepnum = 0
        straight_steps_remaining = 0
        state = self.initial_state
        while (stepnum < self.max_timestep and state not in self.env.terminal_states):
            # Generate next action, state according to parameterized policy
            if (straight_steps_remaining == 0):
                straight_steps_remaining = self.env.control_freq
                action_probs_tuple = self.get_action_probs(state)
                valid_actions = action_probs_tuple[0]
                action_probs = action_probs_tuple[1]
                next_action = self.random_generator.choice(valid_actions, p=action_probs)
                states.append(state)
            state = self.env.get_next_state(next_action, state)
            stepnum += 1
            straight_steps_remaining -= 1
        if (state in self.env.terminal_states):
            states.append(state)
        print(states)