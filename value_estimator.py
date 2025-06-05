import agent, environment
from environment import TerminalStateException, Actions
import random

class MonteCarlo:

    # `environment` is a Gridworld object and `gamma` is the discount factor
    # `policy` is a function mapping from tuples (x,y) to Actions
    def __init__(self, initial_policy, initial_state, environment, gamma):
        self.gamma = gamma
        self.env = environment
        self.visits = dict()
        self.values = dict()
        self.policy = initial_policy
        self.initial_state = initial_state
        self.state = initial_state
        
    def reset_state(self):
        self.state = self.initial_state
        
    # Returns the value function for a given state
    def value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0
        
    def generate_random_action(state):
        valid_actions = list()
        if (not state[0] == 0):
            valid_actions.append(Actions.LEFT)
        if (not state[0] == 9):
            valid_actions.append(Actions.RIGHT)
        if (not state[1] == 0):
            valid_actions.append(Actions.DOWN)
        if (not state[1] == 9):
            valid_actions.append(Actions.UP)
        return random.choice(valid_actions)

    # Generate one episode and update agent's value function
    def episode(self):
        self.reset_state()
        STEP_LIMIT = 1000         # Prevent infinite loops. Is there a better way to do this?
        # states[n]: state at timestep n
        # rewards[n]: reward at timestep n+1
        states = list()
        rewards = list()
        try:
            for i in range(STEP_LIMIT):
                curr_state = self.state
                next_action = self.policy(curr_state)
                states.append(curr_state)
                rewards.append(self.env.reward(next_action, curr_state))
                self.state = self.env.next_state(next_action, curr_state)
        except TerminalStateException:
            # self.visits for the terminal state will always be 0, and value function will also be 0
            i = len(states) - 1
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

    def __init__(self):
        pass