from environment import Actions
import random

class Agent:
    
    # `initial_policy` is a dict mapping from tuples (x,y) representing grid coordinates
    # to members of the enum Actions. `values` is the current state-value function,
    # mapping from tuples (x,y) to floats, and is initially zero for all states
    # Agent's initial state is `initial_state`
    def __init__(self, initial_policy, initial_state):
        self.policy = initial_policy
        self.values = dict()
        self.initial_state = initial_state
        self.state = initial_state

    # Update the entire policy to a new dict `new_policy`
    def update_entire_policy(self, new_policy):
        self.policy = new_policy

    # Updates the policy for one state `s` to the action `a`
    def update_policy(self, s, a):
        self.policy[s] = a
    
    # Returns the value function for a given state
    def value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0

    # Return the next action according to the agent's current policy
    def get_next_action(self):
        return self.policy[self.state]
    
    def reset_state(self):
        self.state = self.initial_state

    def generate_random_action(self):
        x = random.random()
        if (self.state[0] == 0 and self.state[1] == 0):
            if (x > 0.5):
                return Actions.UP
            else:
                return Actions.RIGHT
        elif (self.state[0] == 0 and self.state[1] == 9):
            if (x > 0.5):
                return Actions.DOWN
            else:
                return Actions.RIGHT
        elif (self.state[0] == 9 and self.state[1] == 0):
            if (x > 0.5):
                return Actions.UP
            else:
                return Actions.LEFT
        elif (self.state[0] == 9 and self.state[1] == 9):
            if (x > 0.5):
                return Actions.DOWN
            else:
                return Actions.LEFT
        elif (self.state[0] == 0):
            if (x > 2/3):
                return Actions.UP
            elif (x > 1/3):
                return Actions.RIGHT
            else:
                return Actions.DOWN
        elif (self.state[0] == 9):
            if (x > 2/3):
                return Actions.UP
            elif (x > 1/3):
                return Actions.LEFT
            else:
                return Actions.DOWN
        elif (self.state[1] == 0):
            if (x > 2/3):
                return Actions.LEFT
            elif (x > 1/3):
                return Actions.UP
            else:
                return Actions.RIGHT
        elif (self.state[1] == 9):
            if (x > 2/3):
                return Actions.LEFT
            elif (x > 1/3):
                return Actions.DOWN
            else:
                return Actions.RIGHT
        else:
            if (x > 3/4):
                return Actions.UP
            elif (x > 1/2):
                return Actions.RIGHT
            elif (x > 1/4):
                return Actions.DOWN
            else:
                return Actions.LEFT