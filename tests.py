import numpy as np
from inverted_pendulum import InvertedPendulum
from control_method import QLearningPendulum
import util


class TestEnv(InvertedPendulum):
    def get_reward(self, a, s):
        return s[0] + s[1]

    def action_feature_vector(self, a, s):
        return np.array([s[0], s[1]])


def test_least_squares():
    params = {"mass": 1, "length": 1, "gravity": 1, "damping": 0.01}
    initial_state = np.array([np.pi + 0.1, 0])
    env = TestEnv(
        params=params,
        initial_state=initial_state,
        sim_timestep=1,
        control_timestep=1,
    )
    solver = QLearningPendulum(
        env,
        initial_state=initial_state,
        epsilon=0.2,
        discount_factor=0.98,
        learning_rate=1e-5,
        max_sim_time=1,
    )

    ls_weights, _, _, _ = util.least_squares_fit(
        solver=solver,
        res=10,
        x_min=0,
        x_max=2 * np.pi,
        y_min=-4,
        y_max=4,
        t_min=-1,
        t_max=1,
    )
    print(ls_weights)

#test_least_squares()


x = [1,2,3,4,5]
sum = 0
for i in range(len(x)):
    sum *= 0.1
    sum += x[len(x) - i - 1]
    print(sum)
