from enum import Enum

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
    def __init__(self, rewards, terminalStates):
        self.rewards = rewards
        self.terminalStates = terminalStates

    # Return whether or not taking action `a` from state `s` is valid.
    # a is an invalid action if it would cause the agent to go out of bounds.
    # `a` is a member of the Enum `Actions`, and `s` is the state, 
    # which is a tuple (x,y) representing the grid coordinate.
    def isValidAction(a, s):
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
    def nextState(a, s):
        if (isValidAction(a, s) == False):
            raise IllegalActionException()
        if (s in terminalStates):
            raise TerminalStateException()
        
        if (a == Actions.LEFT):
            s[0] = s[0] - 1
        elif (a == Actions.RIGHT):
            s[0] = s[0] + 1
        elif (a == Actions.UP):
            s[1] = s[1] + 1
        elif (a == Actions.DOWN):
            s[1] = s[1] - 1
        
        return s

    # Reward function
    def reward(self, a, s):
        if (isValidAction(a, s) == False):
            raise IllegalActionException()

        if nextState(a, s) in rewards:
            return rewards.get(s)
        else:
            return 0
    
            



    