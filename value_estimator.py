import math

class MonteCarlo:
    # Every-visit Monte Carlo

    # `policy` is a function mapping from tuples (x,y) to Actions
    def __init__(self, initial_policy, initial_state, environment, discount_factor, max_timestep=1000):
        self.policy = initial_policy
        self.initial_state = initial_state
        self.env = environment
        self.discount_factor = discount_factor
        self.max_timestep = max_timestep # use float('inf') for infinite horizon
        self.visits = dict()
        self.values = dict()
        self.state = initial_state
        
    def reset_state(self):
        self.state = self.initial_state
        
    # Returns the value function for a given state
    def get_state_value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0

    # Generate one episode and update agent's value function
    def run_episode(self):
        self.reset_state()
        states = list()         # states[n]: state at timestep n
        rewards = list()        # rewards[n]: reward at timestep n+1
        stepnum = 0
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            next_action = self.policy(self.state)
            states.append(self.state)
            rewards.append(self.env.get_action_reward(next_action, self.state))
            self.state = self.env.get_next_state(next_action, self.state, self.env.control_freq)
            stepnum += 1

        # Calculate reward of terminal state over remaining timesteps
        if (self.state in self.env.terminal_states):
            states.append(self.state)
            if (self.discount_factor == 1):
                terminal_reward = self.env.get_state_reward(states[-1]) * (self.max_timestep - stepnum)
            elif (math.isinf(self.max_timestep)):
                terminal_reward = self.env.get_state_reward(states[-1]) / (1 - self.discount_factor)
            else:
                terminal_reward = self.env.get_state_reward(states[-1]) * ((1 - (self.discount_factor ** (self.max_timestep - stepnum))) / (1 - self.discount_factor))
            rewards.append(terminal_reward)

        if (stepnum == self.max_timestep):
            rewards.append(self.env.get_state_reward(states[-1]))

        # Update value function
        i = len(states)
        return_i = 0
        while (i > 0):
            i -= 1
            return_i = rewards[i] + self.discount_factor*return_i
            if (not states[i] in self.visits):
                self.visits[states[i]] = 1
                self.values[states[i]] = return_i
            else:
                self.visits[states[i]] += 1
                self.values[states[i]] += (return_i - self.values[states[i]]) / (self.visits[states[i]])

class TDLambda:
    # Online TDLambda

    def __init__(self, initial_policy, initial_state, environment, discount_factor, learning_rate, trace_decay, max_timestep=1000):
        self.policy = initial_policy
        self.initial_state = initial_state
        self.env = environment
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.max_timestep = max_timestep # use float('inf') for infinite horizon
        self.state = self.initial_state
        self.values = dict()
    
    def reset_state(self):
        self.state = self.initial_state
    
    def get_state_value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0
    
    # Generate one episode and update agent's value function
    def run_episode(self):
        self.reset_state()
        eligibility_traces = dict()     # Map from states (tuples) to floats. Eligibility trace of all states not in `eligibility_traces` is 0
        stepnum = 0
        
        while (stepnum < self.max_timestep and self.state not in self.env.terminal_states):
            next_action = self.policy(self.state)
            next_state = self.env.get_next_state(next_action, self.state, self.env.control_freq)
            reward = self.env.get_action_reward(next_action, self.state)
            td_error = reward + self.discount_factor * self.get_state_value(next_state) - self.get_state_value(self.state)
            
            # Update eligibility traces
            for state in eligibility_traces:
                eligibility_traces[state] = self.discount_factor * self.trace_decay * eligibility_traces[state]
            if (self.state in eligibility_traces):
                eligibility_traces[self.state] += 1
            else:
                eligibility_traces[self.state] = 1
            
            # Update value function
            for state in eligibility_traces:
                self.values[state] = self.get_state_value(state) + (self.learning_rate * td_error * eligibility_traces[state])

            self.state = next_state
            stepnum += 1

        # Terminal state reached. This code basically just does the same thing as the previous loop.
        # ? There may be a closed form expression for the value function of each state once a terminal state is reached, given a number of remaining steps. However, for now this explicitly simulates each state
        for i in range(self.max_timestep - stepnum):
            reward = self.env.get_state_reward(self.state)
            td_error = reward + self.discount_factor * self.get_state_value(self.state) - self.get_state_value(self.state)
            
            # Update eligibility traces
            for state in eligibility_traces:
                eligibility_traces[state] = self.discount_factor * self.trace_decay * eligibility_traces[state]
            if (self.state in eligibility_traces):
                eligibility_traces[self.state] += 1
            else:
                eligibility_traces[self.state] = 1
            
            # Update value function
            for state in eligibility_traces:
                self.values[state] = self.get_state_value(state) + (self.learning_rate * td_error * eligibility_traces[state])