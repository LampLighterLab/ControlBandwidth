class QLearning:
    # Implements the Q Learning policy improvement algorithm with a linear value function approximator.
    def __init__(self, env):
        self.env = env
        
    # Compute least-squares approximation
    def update_approximation(self):
        pass
    
    def episode(self):
        pass
    
        

class LinearApprox:

    def __init__(self):
        self.weights = [0] * 7

    # Returns the feature vector (a 7-tuple) for action a and state s
    # [x'x, y'y, x'y, xy', x, y, 1]
    def features(self, a, s):
        x0 = s[0]
        y0 = s[1]
        next_state = self.env.next_state(a, s)
        x1 = next_state[0]
        y1 = next_state[1]
        return (x1*x0, y1*y0, x1*y0, x0*y1, x0, y0, 1)

    def action_value(self, a, s):
        feature_vector = self.features(a, s)
        sum = 0
        for i in range(len(feature_vector)):
            sum += feature_vector[i]*self.weights[i]
        return sum