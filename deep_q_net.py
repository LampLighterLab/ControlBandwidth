import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.module):

    def __init__(self, num_features, torque_res):
        super(DQN, self).__init__()
        self.hidden_layer = nn.Linear(num_features, 8)
        self.output_layer = nn.Linear(8, torque_res)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNPendulum:
    def __init__(
        self,
        env,
        initial_state,
        epsilon,
        discount_factor,
        learning_rate,
        max_sim_time=100,
    ):
        self.env = env
        self.state = initial_state
        self.initial_state = initial_state
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.max_sim_time = max_sim_time

        self.max_torque = (
            0.5
            * self.env.params["mass"]
            * self.env.params["gravity"]
            * self.env.params["length"]
        )

        num_features = self.env.action_feature_vector(0, self.initial_state).shape[0]
        self.policy_net = DQN(num_features=num_features, torque_res=3)
        self.experience_memory = torch.zeros((0,4))

        self.random_generator = np.random.default_rng(123)

    