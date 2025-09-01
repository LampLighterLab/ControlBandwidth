from enum import Enum
import numpy as np


# Set of actions that can be taken
class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Gridworld:
    # Grid coordinates represented by tuples (x,y), where (0,0) is at the bottom left corner
    def __init__(self, rewards, terminal_states, size=(10, 10), control_freq=1):
        self.rewards = (
            rewards  # dict: tuple (x,y) -> scalar, representing nonzero rewards
        )
        self.terminal_states = terminal_states
        self.SIZE_X = size[0]
        self.SIZE_Y = size[1]
        self.control_freq = control_freq
        self.random_generator = np.random.default_rng(54321)

    # Return whether or not taking action `a` in state `s` is valid.
    # a is an invalid action if it would cause the agent to go out of bounds.
    # `a` is a member of Actions, and `s` is the state, which is a tuple (x,y).
    def is_valid_action(self, a, s):
        if s[1] <= 0 and a == Actions.DOWN:
            return False
        elif s[1] >= self.SIZE_Y - self.control_freq and a == Actions.UP:
            return False
        elif s[0] <= 0 and a == Actions.LEFT:
            return False
        elif s[0] >= self.SIZE_X - self.control_freq and a == Actions.RIGHT:
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
    # Do not call when `s` is a terminal state, it should be already known to not update the state in that case
    def get_next_state(self, a, s, step):
        if not self.is_valid_action(a, s):
            return s
        final_state = s
        if a == Actions.LEFT:
            final_state = (s[0] - step, s[1])
        elif a == Actions.RIGHT:
            final_state = (s[0] + step, s[1])
        elif a == Actions.UP:
            final_state = (s[0], s[1] + step)
        elif a == Actions.DOWN:
            final_state = (s[0], s[1] - step)
        final_state = (
            np.clip(final_state[0], 0, self.SIZE_X),
            np.clip(final_state[1], 0, self.SIZE_Y),
        )
        return final_state

    # Returns the reward from ending in state `s`
    def get_state_reward(self, s):
        if s in self.rewards:
            return self.rewards.get(s)
        else:
            return 0

    # Returns the reward from taking action `a` in state `s`
    def get_action_reward(self, a, s):
        if s in self.terminal_states:
            return self.rewards.get(s)
        if not self.is_valid_action(a, s):
            return 0

        final_state = self.get_next_state(a, s, self.control_freq)
        reward_sum = 0
        while s != final_state:
            s = self.get_next_state(a, s, 1)
            reward_sum += self.get_state_reward(s)
        return reward_sum

    # Generate a valid random action
    def generate_random_action(self, state):
        valid_actions = list()
        if not state[0] == 0:
            valid_actions.append(Actions.LEFT)
        if not state[0] == self.SIZE_X - 1:
            valid_actions.append(Actions.RIGHT)
        if not state[1] == 0:
            valid_actions.append(Actions.DOWN)
        if not state[1] == self.SIZE_Y - 1:
            valid_actions.append(Actions.UP)
        return self.random_generator.choice(valid_actions)

    # Feature vector contains one entry for each action-(state observation) pair
    # [(x^2 if a == UP, 0 otherwise), (y^2 if a == UP, 0 otherwise), ... ,
    #  (x^2 if a == RIGHT, 0 otherwise), ... ,
    #  (x^2 if a == DOWN, 0 otherwise), ... ,
    #  (x^2 if a == LEFT, 0 otherwise), ... ]
    @classmethod  # ! make this not classmethod for consistency
    def feature_vector(cls, a, s):
        x = s[0] - 8
        y = s[1] - 8
        i = a.value
        num_state_features = 6
        return np.array(
            [0] * num_state_features * (i)
            + [x**2, y**2, 1, x, y, x * y]
            + [0] * num_state_features * (3 - i)
        )
