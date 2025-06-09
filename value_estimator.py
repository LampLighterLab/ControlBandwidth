import environment
from environment import TerminalStateException, Actions
import random
import math

class MonteCarlo:
    # Every-visit Monte Carlo

    # `environment` is a Gridworld object and `gamma` is the discount factor
    # `policy` is a function mapping from tuples (x,y) to Actions
    def __init__(self, initial_policy, initial_state, environment, gamma, max_timestep=100000):
        self.policy = initial_policy
        self.initial_state = initial_state
        self.env = environment
        self.gamma = gamma
        self.max_timestep = max_timestep # use float('inf') for infinite horizon
        self.visits = dict()
        self.values = dict()
        self.state = initial_state
        
    def reset_state(self):
        self.state = self.initial_state
        
    # Returns the value function for a given state
    def value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0

    # Generate one episode and update agent's value function
    def episode(self):
        self.reset_state()
        states = list()         # states[n]: state at timestep n
        rewards = list()        # rewards[n]: reward at timestep n+1
        stepnum = 0
        try:
            while (stepnum < self.max_timestep):
                curr_state = self.state # TODO: Having both `curr_state` and `self.state` is redundant
                # ? This handles the fact that the random policy function relies on `self` (taking 2 args), and an arbitrary function assigned to `policy` may only take in 1 argument. There may be a better way to do this
                try:
                    next_action = self.policy(self, curr_state)
                except:
                    next_action = self.policy(curr_state)
                states.append(curr_state)
                rewards.append(self.env.action_reward(next_action, curr_state))
                self.state = self.env.next_state(next_action, curr_state)
                stepnum += 1
        except TerminalStateException:
            if (self.gamma == 1):
                terminal_reward = self.env.state_reward(states[-1]) * (self.max_timestep - stepnum)
            elif (math.isinf(self.max_timestep)):
                terminal_reward = self.env.state_reward(states[-1]) / (1 - self.gamma)
            else:
                terminal_reward = self.env.state_reward(states[-1]) * ((1 - (self.gamma ** (self.max_timestep - stepnum))) / (1 - self.gamma))
            rewards.append(terminal_reward)

        if (stepnum == self.max_timestep):
            rewards.append(self.env.state_reward(states[-1]))

        i = len(states)
        return_i = 0
        while (i > 0):
            i -= 1
            return_i = rewards[i] + self.gamma*return_i
            if (not states[i] in self.visits):
                self.visits[states[i]] = 1
                self.values[states[i]] = return_i
            else:
                self.visits[states[i]] += 1
                self.values[states[i]] += (return_i - self.values[states[i]]) / (self.visits[states[i]])

class TDLambda:
    # Online TDLambda

    def __init__(self, initial_policy, initial_state, environment, gamma, alpha, LAMBDA, max_timestep=100000):
        self.policy = initial_policy
        self.initial_state = initial_state
        self.env = environment
        self.gamma = gamma
        self.alpha = alpha
        self.LAMBDA = LAMBDA # can't use variable name "lambda"
        self.max_timestep = max_timestep # use float('inf') for infinite horizon
        self.state = self.initial_state
        self.values = dict()
    
    def reset_state(self):
        self.state = self.initial_state
    
    def value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0
    
    # Generate one episode and update agent's value function
    def episode(self):
        self.reset_state()
        eligibility_traces = dict()     # Map from states (tuples) to floats. Eligibility trace of all states not in `eligibility_traces` is 0
        stepnum = 0
        try:
            while (stepnum < self.max_timestep):
                # ? This handles the fact that the random policy function relies on `self` (taking 2 args), and an arbitrary function assigned to `policy` may only take in 1 argument. There may be a better way to do this
                try:
                    next_action = self.policy(self, self.state)
                except:
                    next_action = self.policy(self.state)
                next_state = self.env.next_state(next_action, self.state) # ! throws terminal state exception, handle this later
                reward = self.env.action_reward(next_action, self.state)
                td_error = reward + self.gamma * self.value(next_state) - self.value(self.state)
                
                # Update eligibility traces
                for state in eligibility_traces:
                    eligibility_traces[state] = self.gamma * self.LAMBDA * eligibility_traces[state]
                if (self.state in eligibility_traces):
                    eligibility_traces[self.state] += 1
                else:
                    eligibility_traces[self.state] = 1
                
                # Update value function
                for state in eligibility_traces:
                    self.values[state] = self.value(state) + (self.alpha * td_error * eligibility_traces[state])

                self.state = next_state
                stepnum += 1
        except TerminalStateException:
            # TODO: Update value function in the case a terminal state is reached. No need to keep simulating steps. Also handle infinite horizon case.
            pass