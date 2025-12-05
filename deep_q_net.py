import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from inverted_pendulum import InvertedPendulum
from animation import get_animation


class DQN(nn.Module):
    def __init__(self, num_features, hidden_layer_size, torque_res):
        super(DQN, self).__init__()
        self.hidden_1 = nn.Linear(num_features, hidden_layer_size)
        self.hidden_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output = nn.Linear(hidden_layer_size, torque_res)

    # Outputs [Q(s, a=-1*max_torque), Q(s, a=0), Q(s, a=max_torque)]
    # or `torque_res` discrete actions but I have it set to 3 for now
    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.output(x)
        return x


def init_normal(model, mean=0.0, std=1.0):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=mean, std=std)
            nn.init.normal_(m.bias, mean=mean, std=std)


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
            0.5  # 1 means the pendulum is just able to hold itself up when parallel to ground
            * self.env.params["mass"]
            * self.env.params["gravity"]
            * self.env.params["length"]
        )

        self.num_features = self.env.state_feature_vector(self.initial_state).shape[0]
        self.features = lambda s: torch.tensor(
            self.env.state_feature_vector(s), dtype=torch.float32
        )
        self.policy_net = DQN(
            num_features=self.num_features, hidden_layer_size=32, torque_res=3
        )
        self.target_net = DQN(
            num_features=self.num_features, hidden_layer_size=32, torque_res=3
        )
        init_normal(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.experience_memory[i, :] contains [s, a, r, s'], length 2*feature vec length + 2
        # description (indices)
        # [features(s) (0:features), action indices (features), r (features+1), features(s') (features+2:2*features+2)]
        self.experience_memory = torch.zeros((0, 2 * self.num_features + 2))

        self.random_generator = np.random.default_rng(1234)

    # Returns the action index of the epsilon-greedy action in linspace over possible actions
    # note DQN returns a vector of length 3 representing the Q values from -1*max torque, 0 torque, max torque
    # this returns the index of the predicted best value [0...2]
    def get_epsilon_greedy_action(self, state, epsilon):
        r = self.random_generator.random()
        if r < epsilon:
            return int(np.floor(self.random_generator.random() * 3))
        return torch.argmax(self.policy_net.forward(self.features(state)))

    # Generate episodes, append transitions from `num_episodes` episodes to the experience memory
    # also return average reward per episode
    def sample_experience(self, num_episodes):
        transition_stack = torch.zeros((0, 2 * self.num_features + 2))
        num_timesteps = int(self.max_sim_time // self.env.control_timestep) - 1
        rewards = torch.zeros((num_episodes, num_timesteps))
        for eps in range(num_episodes):
            # Reset states
            curr_state = torch.tensor(self.initial_state + 0.5 * np.random.random(2), dtype=torch.float32)
            next_state = torch.tensor(self.initial_state)
            for timestep in range(num_timesteps):
                action_index = self.get_epsilon_greedy_action(
                    state=curr_state, epsilon=self.epsilon
                )
                torques = torch.linspace(-1 * self.max_torque, self.max_torque, 3)
                reward = self.env.get_reward(torques[action_index], curr_state)
                reward = torch.tensor([reward])
                rewards[eps, timestep] = reward  # keep track of rewards
                next_state_np = self.env.get_next_state(
                    torques[action_index].numpy(), curr_state.numpy()
                )
                next_state = torch.from_numpy(next_state_np)
                action_index = torch.tensor([action_index])
                transition = torch.cat(
                    (
                        self.features(curr_state),
                        action_index,
                        reward,
                        self.features(next_state),
                    )
                )
                curr_state = torch.tensor(next_state).detach()
                transition_stack = torch.cat(
                    (transition_stack, transition.t().reshape(1, -1)), dim=0
                )
            print(f"Episode {eps} stored in experience memory")
        self.experience_memory = torch.cat(
            (self.experience_memory, transition_stack), dim=0
        )
        return torch.mean(rewards, dim=1)

    # Remove the transitions from the `num_episodes` oldest episodes in the experience memory
    def clear_experience_memory(self, num_episodes):
        num_transitions = np.min(
            [
                self.experience_memory.shape[0],
                num_episodes * int(self.max_sim_time // self.env.control_timestep - 1),
            ]
        )
        self.experience_memory = self.experience_memory[num_transitions:, :]
        print(f"{num_transitions} transitions removed from experience memory")

    # Perform GD using Pytorch on entire experience memory, and periodically copy policy net weights to target net
    # return loss for each epoch
    def train_policy_net(self, num_epochs):
        optimizer = optim.SGD(
            self.policy_net.parameters(), lr=self.learning_rate, momentum=0.8
        )
        epoch_losses = torch.zeros(num_epochs)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            action_indices = self.experience_memory[:, self.num_features]
            target_net_preds = torch.max(
                self.target_net.forward(
                    self.experience_memory[:, self.num_features + 2 :]
                )
            )
            policy_net_preds_all_actions = self.policy_net.forward(
                self.experience_memory[:, 0 : self.num_features]
            )  # n*3
            policy_net_preds = torch.zeros(
                (self.experience_memory.shape[0])
            )  # 1d array of length n
            for i in range(policy_net_preds_all_actions.shape[0]):
                policy_net_preds[i] = policy_net_preds_all_actions[
                    i, int(action_indices[i])
                ]
            rewards = self.experience_memory[:, self.num_features + 1]
            loss_func = nn.MSELoss()
            loss = loss_func(
                rewards + self.discount_factor * target_net_preds, policy_net_preds
            )
            loss.backward()
            optimizer.step()
            epoch_losses[epoch] = loss.item()
            print("Epoch {} MSE loss is {}".format(epoch + 1, loss.item()))

            if epoch % 10 == 9:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print("policy net weights copied to target net")
        return epoch_losses


def sample_trajectory(solver):
    env = solver.env
    initial_state = np.array(solver.initial_state)
    num_timesteps = int(solver.max_sim_time // solver.env.sim_timestep)
    state_traj = np.zeros((initial_state.size, num_timesteps + 1))
    action_traj = np.zeros(num_timesteps + 1)
    state_traj[:, 0] = initial_state
    solver.state = solver.initial_state
    torques = np.linspace(-1 * solver.max_torque, solver.max_torque, 3)
    for i in range(num_timesteps):
        next_action_index = solver.get_epsilon_greedy_action(solver.state, epsilon=0)
        action_traj[i] = torques[next_action_index]
        # set first argument to 0 to test without control
        next_state = env.get_next_state(action_traj[i], state_traj[:, i])
        state_traj[:, i + 1] = next_state
        solver.state = next_state
    state_traj[0, :] = np.mod(state_traj[0], 2 * np.pi)
    return state_traj, action_traj


# Initialize DQN object
env = InvertedPendulum(
    params={"length": 1, "mass": 1, "gravity": 1, "damping": 0.01},
    initial_state=[0.1, 0.0],
    sim_timestep=0.005,
    control_timestep=0.05,
)
p = DQNPendulum(
    env=env,
    initial_state=[0.1, 0.0],
    epsilon=0.2,
    discount_factor=0.98,
    learning_rate=1e-7,
    max_sim_time=10,
)

# training
avg_rewards = p.sample_experience(num_episodes=20)
epoch_losses = p.train_policy_net(num_epochs=200)
# p.clear_experience_memory(num_episodes=50)

# get sample trajectory
sample_state_traj, sample_action_traj = sample_trajectory(p)
sample_state_traj = np.array(sample_state_traj)
sample_action_traj = np.array(sample_action_traj)

# make plots, create sample trajectory video
num_episode = avg_rewards.shape[0]
num_epoch = epoch_losses.shape[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

ax1.plot(range(num_episode), avg_rewards)
ax1.set_title("Average reward per episode")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Average reward")
ax1.grid(True)

ax2.plot(range(num_epoch), epoch_losses)
ax2.set_title("Training loss per epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE Loss")
ax2.grid(True)

plt.tight_layout()
plt.savefig("out/DQN_rewards_losses.png")
print("Rewards and losses figures saved to out/DQN_rewards_losses.png")

get_animation(
    sample_state_traj, sample_action_traj, filename="out/dqn_sample_traj_animation.mp4"
)
