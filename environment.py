from enum import Enum
import random

# Set of actions that can be taken. The assigned values are meaningless
class Actions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

# Should not occur if everything goes right
class IllegalActionException(Exception):
    def __init__(self):
        super().__init__("Attempted action is invalid for this state")

class Gridworld:
    # Immutable class

    # Constructor will take in a dict `rewards` mapping from a tuple (x,y) to a float,
    # representing coordinates of cells whose rewards are nonzero. `terminalStates` is a list of
    # tuples (x,y) which represent coordinates of terminal states.
    # x will represent the horizontal position (0 on the left), and y will represent
    # the vertical position (0 on the bottom)
    def __init__(self, rewards, terminal_states, size=(10,10), control_freq=1):
        self.rewards = rewards
        self.terminal_states = terminal_states
        self.SIZE_X = size[0]
        self.SIZE_Y = size[1]
        self.control_freq = control_freq

    # Return whether or not taking action `a` in state `s` is valid.
    # a is an invalid action if it would cause the agent to go out of bounds.
    # `a` is a member of Actions, and `s` is the state, which is a tuple (x,y).
    def is_valid_action(self, a, s):
        if (s[1] == 0 and a == Actions.DOWN):
            return False
        elif (s[1] == self.SIZE_Y - 1 and a == Actions.UP):
            return False
        elif (s[0] == 0 and a == Actions.LEFT):
            return False
        elif (s[0] == self.SIZE_X - 1 and a == Actions.RIGHT):
            return False
        else:
            return True
    
    # Get a list of all valid actions from state `s`
    def get_valid_actions(self, s):
        a = list()
        for action in Actions:
            if self.is_valid_action(action, s):
                a.append(action)
        return a

    # Return the new state s' from taking action `a` in state `s`
    # Do not call when `s` is a terminal state.
    def get_next_state(self, a, s):
        if (not self.is_valid_action(a, s)):
            return s
        
        if (a == Actions.LEFT):
            return (s[0]-1, s[1])
        elif (a == Actions.RIGHT):
            return (s[0]+1, s[1])
        elif (a == Actions.UP):
            return (s[0], s[1]+1)
        elif (a == Actions.DOWN):
            return (s[0], s[1]-1)

    # Returns the reward from taking any action ending in state `s`
    def get_state_reward(self, s):
        if s in self.rewards:
            return self.rewards.get(s)
        else:
            return 0

    # Returns the reward from taking action `a` in state `s`
    def get_action_reward(self, a, s):
        if (s in self.terminal_states):
            return self.rewards.get(s)
        if (not self.is_valid_action(a, s)):
            return 0

        next_state = self.get_next_state(a, s)
        return self.get_state_reward(next_state)

    # Generate a valid random action
    def generate_random_action(self, state):
        valid_actions = list()
        if (not state[0] == 0):
            valid_actions.append(Actions.LEFT)
        if (not state[0] == self.SIZE_X - 1):
            valid_actions.append(Actions.RIGHT)
        if (not state[1] == 0):
            valid_actions.append(Actions.DOWN)
        if (not state[1] == self.SIZE_Y - 1):
            valid_actions.append(Actions.UP)
        return random.choice(valid_actions)