import agent, environment
from environment import TerminalStateException

class MonteCarlo:

    # `agent` is an Agent object, `environment` is a Gridworld object,
    # and `gamma` is the discount factor
    def __init__(self, agent, environment, gamma):
        self.agent = agent
        self.gamma = gamma
        self.env = environment
        self.visits = dict()

    # Returns the return value from a list `seq` of rewards, starting at the nth item in `seq`
    def return_from_sequence(self, seq, n):
        if (n == len(seq)):
            return 0
        return_val = seq[n]
        discount_factor = 1
        for i in range(n+1, len(seq)):
            discount_factor *= self.gamma
            return_val += seq[i]*discount_factor
        return return_val

    # Generate one episode and update agent's value function
    def episode(self):
        self.agent.reset_state()
        STEP_LIMIT = 100000         # Prevent infinite loops. Is there a better way to do this?
        # states[n]: state at timestep n
        # rewards[n]: reward at timestep n+1
        states = list()
        rewards = list()
        try:
            for i in range(STEP_LIMIT):
                curr_state = self.agent.state
                next_action = self.agent.get_next_action()
                states.append(curr_state)
                rewards.append(self.env.reward(next_action, curr_state))
                self.agent.state = self.env.next_state(next_action, curr_state)
        except TerminalStateException:
            for i in range(len(states)):
                # this runs in O(n^2) time with the length of the episode. I think there is a way to improve this
                episode_return = self.return_from_sequence(rewards, i)
                if (not states[i] in self.visits):
                    self.visits[states[i]] = 1
                    self.agent.values[states[i]] = episode_return
                else:
                    self.visits[states[i]] = self.visits[states[i]] + 1
                    self.agent.values[states[i]] += (episode_return - self.agent.values[states[i]])*(1/self.visits[states[i]])

class TDLambda:

    def __init__(self):
        pass