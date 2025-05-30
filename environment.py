from enum import Enum
import random

# Set of actions that can be taken. The assigned values are meaningless
class Actions(Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class IllegalActionException(Exception):
    def __init__(self):
        super().__init__("Attempted action is invalid for this state")

class TerminalStateException(Exception):
    def __init__(self):
        super().__init__("Terminal state reached")

class Gridworld:
    # Immutable class

    # Constructor will take in a dict `rewards` mapping from a tuple (x,y) to a float,
    # representing coordinates of cells whose rewards are nonzero. `terminalStates` is a list of
    # tuples (x,y) which represent coordinates of terminal states.
    # x will represent the horizontal position (0 on the left), and y will represent
    # the vertical position (0 on the bottom)
    # Size of the grid world is fixed at 10x10 for now.
    def __init__(self, rewards, terminal_states):
        self.rewards = rewards
        self.terminal_states = terminal_states

    # Return whether or not taking action `a` in state `s` is valid.
    # a is an invalid action if it would cause the agent to go out of bounds.
    # `a` is a member of the Enum `Actions`, and `s` is the state, 
    # which is a tuple (x,y) representing the grid coordinate.
    def is_valid_action(self, a, s):
        if (s[1] == 0 and a == Actions.DOWN):
            return False
        elif (s[1] == 9 and a == Actions.UP):
            return False
        elif (s[0] == 0 and a == Actions.LEFT):
            return False
        elif (s[0] == 9 and a == Actions.RIGHT):
            return False
        else:
            return True 

    # Return the new state s' from taking action `a` in state `s`, or throw
    # IllegalActionException if the action is invalid. Throw TerminalStateException
    # if attempting to perform an action from a terminal state.
    def next_state(self, a, s):
        if (s in self.terminal_states):
            raise TerminalStateException()
        if (not self.is_valid_action(a, s)):
            raise IllegalActionException()
        
        if (a == Actions.LEFT):
            return (s[0]-1, s[1])
        elif (a == Actions.RIGHT):
            return (s[0]+1, s[1])
        elif (a == Actions.UP):
            return (s[0], s[1]+1)
        elif (a == Actions.DOWN):
            return (s[0], s[1]-1)

    # Returns the reward from taking action `a` in state `s`
    def reward(self, a, s):
        if (s in self.terminal_states):
            raise TerminalStateException()
        if (not self.is_valid_action(a, s)):
            raise IllegalActionException()

        next_state = self.next_state(a, s)
        if next_state in self.rewards:
            return self.rewards.get(next_state)
        else:
            return 0