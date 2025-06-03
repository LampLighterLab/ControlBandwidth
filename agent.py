from environment import Actions
import random

class Agent:
    
    # `initial_policy`: dict (x,y) -> Action OR function(any) -> Action
    # `values` is the current state-value function,
    # mapping from tuples (x,y) to floats, and is initially zero for all states
    # Agent's initial state is `initial_state`
    def __init__(self, initial_policy, initial_state):
        self.policy = initial_policy
        self.values = dict()
        self.initial_state = initial_state
        self.state = initial_state

    def update_policy(self, new_policy):
        self.policy = new_policy

    # Updates the policy for one state `s` to the action `a`. Requires self.policy is a dict
    def update_policy_dict_entry(self, s, a):
        self.policy[s] = a
    
    # Returns the value function for a given state
    def value(self, s):
        if (s in self.values):
            return self.values.get(s)
        else:
            return 0

    # Return the next action according to the agent's current policy
    def get_next_action(self):
        try:
            return self.policy[self.state]
        except:
            return self.policy()
    
    def reset_state(self):
        self.state = self.initial_state

    def generate_random_action(self):
        valid_actions = list()
        if (not self.state[0] == 0):
            valid_actions.append(Actions.LEFT)
        if (not self.state[0] == 9):
            valid_actions.append(Actions.RIGHT)
        if (not self.state[1] == 0):
            valid_actions.append(Actions.DOWN)
        if (not self.state[1] == 9):
            valid_actions.append(Actions.UP)
        return random.choice(valid_actions)