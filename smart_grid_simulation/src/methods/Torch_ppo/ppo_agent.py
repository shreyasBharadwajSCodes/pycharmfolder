import torch
import torch.optim as optim
from policy_network import PolicyNetwork


class PPOAgent:
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim, lr=1e-4, gamma=0.99, epsilon=0.2,
                 beta=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta

        self.policy_net = PolicyNetwork(state_dim, discrete_action_dim, continuous_action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, action_means, value = self.policy_net(state)

        action_dist_discrete = torch.distributions.Categorical(action_probs)
        action_discrete = action_dist_discrete.sample().unsqueeze(-1).float()

        action_dist_continuous = torch.distributions.Normal(action_means, torch.ones_like(action_means))
        action_continuous = action_dist_continuous.sample()

        log_prob_discrete = action_dist_discrete.log_prob(action_discrete.squeeze(-1))
        log_prob_continuous = action_dist_continuous.log_prob(action_continuous).sum(dim=-1)

        # Ensure both actions have the same number of dimensions
        action_discrete = action_discrete.unsqueeze(0) if action_discrete.dim() == 1 else action_discrete
        action_continuous = action_continuous.unsqueeze(0) if action_continuous.dim() == 1 else action_continuous
        action = torch.cat([action_discrete, action_continuous], dim=-1).squeeze()
        log_prob = log_prob_discrete + log_prob_continuous

        return action.cpu().numpy(), log_prob, value

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            discounted_reward = reward + self.gamma * discounted_reward * mask
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)
        masks = torch.cat(self.masks)
        returns = rewards
        advantage = returns - values

        self.optimizer.zero_grad()

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        entropy = -(log_probs * torch.exp(log_probs)).mean()

        loss = actor_loss + 0.5 * critic_loss - self.beta * entropy
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
