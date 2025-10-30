from environment import Actions
import numpy as np
from numpy.linalg import LinAlgError
import math
from environment import Gridworld
from inverted_pendulum import InvertedPendulum


class QLearning:
    # Implements the Q Learning policy improvement algorithm with action-value function approximation

    def __init__(self, env, initial_state, epsilon, discount_factor, max_sim_time=1000):
        self.env = env
        self.initial_state = initial_state
        self.epsilon = (
            epsilon  # Random exploration factor in epsilon-greedy behavior policy
        )
        self.discount_factor = discount_factor
        self.max_sim_time = max_sim_time
        self.state = initial_state
        self.random_generator = np.random.default_rng(1234526)

        self.weights = np.zeros(
            len(Gridworld.feature_vector(Actions.UP, initial_state))
        )

    def action_value_approx(self, a, s):
        return np.dot(self.weights, Gridworld.feature_vector(a, s))

    # Find least squares solution for weights using TD(0) error
    # states, rewards are lists of their values in an episode
    def update_weights(self, states, rewards, actions):
        outer_prod = np.zeros((len(self.weights), len(self.weights)))
        r_vec = np.zeros(len(self.weights))
        for i in range(len(states) - 1):
            a = np.array(Gridworld.feature_vector(actions[i], states[i]))
            b = np.array(
                Gridworld.feature_vector(actions[i], states[i])
                - np.multiply(
                    self.discount_factor,
                    Gridworld.feature_vector(actions[i + 1], states[i + 1]),
                )
            )
            a = np.reshape(a, (a.shape[0], 1))
            b = np.reshape(b, (1, b.shape[0]))
            outer_prod = np.add(outer_prod, a * b)
        for i in range(len(states)):
            r_vec = np.add(
                r_vec,
                np.multiply(
                    rewards[i], Gridworld.feature_vector(actions[i], states[i])
                ),
            )
        # "cheating" to get around non-invertibility
        try:
            outer_prod = np.linalg.inv(outer_prod)
        except LinAlgError:
            outer_prod = np.add(
                outer_prod, np.multiply(0.0001, np.identity(len(self.weights)))
            )
            outer_prod = np.linalg.inv(outer_prod)
        self.weights = np.matmul(outer_prod, r_vec)

    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        next_action = (
            Actions.UP
        )  # declared here to keep `next_action` in scope. Initial action is arbitrary
        states = list()
        rewards = list()
        actions = list()
        while (
            stepnum < self.max_sim_time and self.state not in self.env.terminal_states
        ):
            # Generate next action, state according to epsilon-greedy policy
            max_q_val = self.action_value_approx(next_action, self.state)
            max_action = next_action
            for action in self.env.get_valid_actions(self.state):
                if self.action_value_approx(action, self.state) > max_q_val:
                    max_q_val = self.action_value_approx(action, self.state)
                    max_action = action
            r = self.random_generator.random()
            if r < self.epsilon:
                next_action = self.env.generate_random_action(self.state)
            else:
                next_action = max_action
            states.append(self.state)
            rewards.append(self.env.get_action_reward(next_action, self.state))
            actions.append(next_action)
            next_state = self.env.get_next_state(
                next_action, self.state, self.env.control_freq
            )

            self.state = next_state
            stepnum += 1
        self.update_weights(
            states, rewards, actions
        )  # ! this resets weights to 0 if the agent ever is timed out

    def get_optimal_path(self):
        path = list()
        curr_state = self.initial_state
        while curr_state not in path and curr_state not in self.env.terminal_states:
            path.append(curr_state)
            max_action = self.env.get_valid_actions(curr_state)[0]
            max_q_val = self.action_value_approx(max_action, curr_state)
            for action in self.env.get_valid_actions(curr_state):
                if self.action_value_approx(action, curr_state) > max_q_val:
                    max_q_val = self.action_value_approx(action, curr_state)
                    max_action = action
            curr_state = self.env.get_next_state(max_action, curr_state, 1)
        if curr_state in path:
            print("Greedy path is cyclic! No path can be found with current Q(s,a)")
        if curr_state in self.env.terminal_states:
            path.append(curr_state)
            print("Path:")
            print(path)
            print(f"Path length is {len(path)}")


class QLearningPendulum:
    # Q learning algorithm for the inverted pendulum, and approximating V(s) as a linear combination of state features

    def __init__(
        self,
        env,
        initial_state,
        epsilon,
        discount_factor,
        learning_rate,
        max_sim_time=100,
    ):
        self.env = env
        self.state = initial_state
        self.initial_state = initial_state
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.max_sim_time = max_sim_time

        self.max_torque = (
            0.5
            * self.env.params["mass"]
            * self.env.params["gravity"]
            * self.env.params["length"]
        )
        self.weights = np.random.rand(
            self.env.action_feature_vector(0, self.initial_state).shape[0]
        )
        self.random_generator = np.random.default_rng(123)
        self.episode_count = 0

    def action_value_approx(self, a, s):
        return np.dot(self.weights, self.env.action_feature_vector(a, s))

    # Generate next action, state according to epsilon-greedy policy
    def get_epsilon_greedy_action(self, state, epsilon):
        # max_torque is the torque needed to hold the pendulum in static equilibrium
        # when the arm is parallel to the ground, multiplied by a constant
        # This allows the controller to exert any torque from -max_torque to max_torque

        r = self.random_generator.random()
        if r < epsilon:
            return (
                self.random_generator.random() * 2 * self.max_torque - self.max_torque
            )

        sample_torques = np.linspace(-1 * self.max_torque, self.max_torque, 9)
        sample_q_vals = np.zeros(sample_torques.size)
        for i in range(sample_torques.size):
            sample_q_vals[i] = self.action_value_approx(
                sample_torques[i], state
            )  # maybe more efficient with vectorized function
        return sample_torques[np.argmax(sample_q_vals)]

    # Return list of states
    def run_episode(self):
        self.state = self.initial_state.copy()

        t_max = int(self.max_sim_time // self.env.control_timestep)
        # states[:, t] is the state [pos, vel] at timestep t
        states = np.zeros((self.initial_state.size, t_max + 1))
        rewards = np.zeros((t_max,))
        actions = np.zeros((t_max,))
        states[:, 0] = self.initial_state.copy()
        for t in range(t_max):
            actions[t] = self.get_epsilon_greedy_action(
                states[:, t], epsilon=self.epsilon
            )
            states[:, t + 1] = self.env.get_next_state(actions[t], states[:, t])
            rewards[t] = self.env.get_reward(actions[t], states[:, t + 1])
            # Update weights: Q(s,a) <- alpha*(R(s,a) + gamma*max(Q(s',a')) - Q(s,a))
            # Get max Q(s',a') using Q-value approx
            max_q_next_action = np.dot(
                self.weights,
                self.env.action_feature_vector(
                    self.get_epsilon_greedy_action(states[:, t + 1], epsilon=0),
                    states[:, t + 1],
                ),
            )
            q_error = (
                rewards[t]
                + self.discount_factor * max_q_next_action
                - np.dot(
                    self.weights,
                    self.env.action_feature_vector(actions[t], states[:, t]),
                )
            )
            self.weights = (
                self.weights
                + self.learning_rate
                * q_error
                * self.env.action_feature_vector(actions[t], states[:, t])
            )
        self.episode_count += 1
        return states


class ReinforceSoftmax:
    # REINFORCE algorithm using the softmax policy

    def __init__(
        self,
        env,
        initial_state,
        discount_factor,
        step_size,
        temperature,
        max_sim_time=1000,
    ):
        self.env = env
        self.initial_state = initial_state
        self.state = initial_state
        self.discount_factor = discount_factor
        self.step_size = step_size
        self.temperature = temperature
        self.max_sim_time = max_sim_time

        self.weights = np.zeros(len(env.feature_vector(initial_state)))

        self.random_generator = np.random.default_rng(123456)

    # Return the set of valid actions and their probabilities according to policy
    def get_action_probs(self, state):
        valid_actions = self.env.get_valid_actions(state)
        action_probs = list()
        sum_weights = 0
        for action in valid_actions:
            feature_scalar = np.dot(
                self.weights, Gridworld.feature_vector(action, state)
            )
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
            gradient_log_policy = np.subtract(
                gradient_log_policy,
                np.multiply(
                    action_probs[i], Gridworld.feature_vector(valid_actions[i], state)
                ),
            )
        return gradient_log_policy

    def run_episode(self):
        self.state = self.initial_state
        stepnum = 0
        states = list()
        actions = list()
        rewards = list()
        while (
            stepnum < self.max_sim_time and self.state not in self.env.terminal_states
        ):
            # Generate next action, state according to parameterized policy
            action_probs_tuple = self.get_action_probs(self.state)
            valid_actions = action_probs_tuple[0]
            action_probs = action_probs_tuple[1]
            next_action = self.random_generator.choice(valid_actions, p=action_probs)
            states.append(self.state)
            actions.append(next_action)
            rewards.append(self.env.get_action_reward(next_action, self.state))
            self.state = self.env.get_next_state(
                next_action, self.state, self.env.control_freq
            )
            stepnum += 1

        # Handle terminal state
        if self.state in self.env.terminal_states:
            states.append(self.state)

        # Calculate returns and update weights
        i = len(states) - 1
        return_i = 0
        while i > 0:
            i -= 1
            return_i = rewards[i] + self.discount_factor * return_i
            self.weights = np.add(
                self.weights,
                self.step_size * return_i * self.score_function(actions[i], states[i]),
            )

    # Generate a sample path without updating weights
    def get_path(self):
        states = list()
        stepnum = 0
        state = self.initial_state
        while stepnum < self.max_sim_time and state not in self.env.terminal_states:
            # Generate next action, state according to parameterized policy
            action_probs_tuple = self.get_action_probs(state)
            valid_actions = action_probs_tuple[0]
            action_probs = action_probs_tuple[1]
            next_action = self.random_generator.choice(valid_actions, p=action_probs)
            states.append(state)
            state = self.env.get_next_state(next_action, state, 1)
            stepnum += 1
        if state in self.env.terminal_states:
            states.append(state)
        print(states)
