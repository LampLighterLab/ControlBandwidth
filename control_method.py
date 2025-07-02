from environment import Actions
import random
import numpy as np
from numpy.linalg import LinAlgError

class QLearning:
    # Implements the Q Learning policy improvement algorithm with state-value function approximation

    def __init__(self, env, initial_state, epsilon, discount_factor, step_size, obs_func, max_timestep=10000):
        self.env = env
        self.initial_state = initial_state
        self.epsilon = epsilon               # Random exploration factor in epsilon-greedy behavior policy
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.max_timestep = max_timestep
        self.state = initial_state
        self.action_value_dict = dict()      # map: (action, (x,y)) -> float
        self.random_generator = np.random.default_rng(1234526)
        
        self.obs_vector = obs_func
        self.weights = [0] * len(obs_func(initial_state))
    
    def state_value_approx(self, s):
        sum = 0
        obs_s = self.obs_vector(s)
        for i in range(len(self.weights)):
            sum += self.weights[i] * obs_s[i]
        return sum
    
    def action_value_approx(self, a, s):
        next_state = self.env.next_state(a, s)
        return self.env.state_reward(s) + self.discount_factor*self.env.state_reward(next_state)
    
    # Find least squares solution for weights using TD(0) error
    # states, rewards are lists of their values in an episode
    def update_weights(self, states, rewards):
        np.set_printoptions(suppress=True, precision=5)

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
                straight_steps_left = self.env.control_freq
                max_q_val = self.action_value_approx(next_action, self.state)
                max_action = next_action
                for action in self.env.valid_actions(self.state):
                    if (self.action_value_approx(action, self.state) > max_q_val):
                        max_q_val = self.action_value_approx(action, self.state)
                        max_action = action
                r = self.random_generator.random()
                if (r < self.epsilon):
                    next_action = self.env.generate_random_action(self.state)
                else:
                    next_action = max_action
            next_state = self.env.get_next_state(next_action, self.state)
            
            states.append(self.state)
            rewards.append(self.env.action_reward(next_action, self.state))
            self.state = next_state
            stepnum += 1
            straight_steps_left -= 1
        self.update_weights(states, rewards)
        
        # Handle terminal state
    
    def get_optimal_path(self):
        path = list()
        curr_state = self.initial_state
        while (curr_state not in path and curr_state not in self.env.terminal_states):
            path.append(curr_state)
            random_action = self.random_generator.choice(self.env.valid_actions(curr_state))
            max_q_val = self.action_value(random_action, curr_state)
            max_action = random_action
            for action in self.env.valid_actions(curr_state):
                if (self.action_value(action, curr_state) > max_q_val):
                    max_q_val = self.action_value(action, curr_state)
                    max_action = action
            curr_state = self.env.get_next_state(max_action, curr_state)
        if curr_state in path:
            print("Greedy path is cyclic! No path can be found with current Q(s,a)")
        if curr_state in self.env.terminal_states:
            path.append(curr_state)
            print("Path:")
            print(path)
            print(f"Path length is {len(path)}")
