import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        super(PolicyNetwork, self).__init__()
        assert isinstance(discrete_action_dim, int), "discrete_action_dim must be an integer"
        assert isinstance(continuous_action_dim, int), "continuous_action_dim must be an integer"

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.actor_discrete = nn.Linear(256, discrete_action_dim)
        self.actor_continuous = nn.Linear(256, continuous_action_dim)

        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        action_probs = torch.softmax(self.actor_discrete(x), dim=-1)
        action_means = self.actor_continuous(x)
        state_value = self.critic(x)

        return action_probs, action_means, state_value
