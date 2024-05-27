import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from models.smart_grid_system import SmartGridSystem

# PPOAgent Class
class PPOAgent:
    def __init__(self, policy_net, lr=1e-4, gamma=0.99, epsilon=0.2, beta=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        if isinstance(self.policy_net, DiscretePolicyNetwork):
            action_probs, value = self.policy_net(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action.item(), log_prob, value
        elif isinstance(self.policy_net, ContinuousPolicyNetwork):
            action_means, value = self.policy_net(state)
            action_dist = torch.distributions.Normal(action_means, torch.ones_like(action_means))
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)

            # Ensure continuous actions are non-negative and within valid bounds
            action = torch.clamp(action, min=0)

            return action.squeeze().cpu().numpy(), log_prob, value

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, mask in zip(reversed(self.rewards), reversed(self.masks)):
            discounted_reward = reward + self.gamma * discounted_reward * mask
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards).squeeze()
        log_probs = torch.cat(self.log_probs).squeeze()
        values = torch.cat(self.values).squeeze()
        masks = torch.FloatTensor(self.masks).squeeze()  # Ensure masks is converted to a FloatTensor
        returns = rewards
        advantage = returns - values

        # Ensure shapes match
        assert log_probs.shape == advantage.shape, f"Shapes of log_probs {log_probs.shape} and advantage {advantage.shape} do not match."

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

# DiscretePolicyNetwork Class
class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_means = self.actor(x)
        state_value = self.critic(x)
        return action_means, state_value

# CustomSmartGridEnv Class
class CustomSmartGridEnv:
    def __init__(self, smart_grid_system):
        self.smart_grid_system = smart_grid_system
        self.time_step = 0

    def reset(self):
        self.smart_grid_system.reset()
        self.time_step = 0
        return self._get_state()

    def step(self, battery_action, solar_action, chp_action, market_action):
        battery_action = int(battery_action)
        solar_action = int(solar_action)

        if isinstance(chp_action, np.ndarray):
            chp_action_value = chp_action[0]
        else:
            chp_action_value = chp_action

        if isinstance(market_action, np.ndarray):
            market_action_value = market_action[0]
        else:
            market_action_value = market_action

        market_action_value = max(0, market_action_value)

        if chp_action_value < self.smart_grid_system.chp.min_operational_value and chp_action_value != 0:
            chp_action_value = self.smart_grid_system.chp.min_operational_value
        elif chp_action_value > self.smart_grid_system.chp.max_operational_value:
            chp_action_value = self.smart_grid_system.chp.max_operational_value

        self._apply_battery_action(battery_action)
        self._apply_solar_action(solar_action)
        self._apply_chp_action(chp_action_value)
        self._apply_market_purchase_action(market_action_value)

        costs = self._calculate_costs()
        penalties = self._calculate_penalties()
        reward = -(costs + penalties)

        reward += self._calculate_market_reward(market_action_value)

        print(f"Timestep: {self.time_step}")
        print(f"Battery Action: {battery_action}, Solar Action: {solar_action}, CHP Action: {chp_action_value}, Market Action: {market_action_value}")
        print(f"Costs: {costs}, Penalties: {penalties}, Reward: {reward}")

        self.time_step += 1
        done = self.time_step >= len(self.smart_grid_system.final_df)
        next_state = self._get_state()

        return next_state, reward, done

    def _apply_battery_action(self, action):
        if action == 0:
            self.smart_grid_system.battery.set_mode("idle")
        elif action == 1:
            self.smart_grid_system.battery.set_mode("charge")
            available_energy = sum([
                self.smart_grid_system.solar_model.get_state(self.time_step)[0],
                self.smart_grid_system.chp.current_output,
                self.smart_grid_system.ppm.get_state(self.time_step),
                self.smart_grid_system.em.get_purchased_units()[self.time_step // 15 if self.time_step // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
            ])
            self.smart_grid_system.battery.charge(available_energy)
        elif action == 2:
            self.smart_grid_system.battery.set_mode("discharge")
            total_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[min(self.time_step, len(self.smart_grid_system.final_df) - 1)]
            supplied_energy = sum([
                self.smart_grid_system.chp.current_output,
                self.smart_grid_system.solar_model.get_state(self.time_step)[0],
                self.smart_grid_system.ppm.get_state(self.time_step),
                self.smart_grid_system.em.get_purchased_units()[self.time_step // 15 if self.time_step // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
            ])
            demand_left = total_demand - supplied_energy
            self.smart_grid_system.battery.discharge(demand_left)

    def _apply_solar_action(self, action):
        if action == 0:
            self.smart_grid_system.solar_model.set_mode("off")
        elif action == 1:
            self.smart_grid_system.solar_model.set_mode("on")

    def _apply_chp_action(self, action):
        self.smart_grid_system.chp.set_output(action)

    def _apply_market_purchase_action(self, action):
        if 6 <= self.time_step % 15 <= 9:
            units = max(0, action)
            self.smart_grid_system.em.make_purchase(units)

    def _calculate_costs(self):
        timestep = min(self.time_step, len(self.smart_grid_system.final_df) - 1)
        battery_cost = self.smart_grid_system.battery.get_cost()
        chp_cost = self.smart_grid_system.chp.calculate_cost_at_current_step()
        market_cost = self.smart_grid_system.em.get_price(timestep) * self.smart_grid_system.em.get_purchased_units()[
            timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
        prior_purchased_cost = self.smart_grid_system.ppm.get_price(self.smart_grid_system.ppm.get_state(timestep))
        solar_cost = self.smart_grid_system.solar_model.calculate_cost_at_timestep(timestep)
        public_grid_cost = self.smart_grid_system.pg.get_price(self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep], sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(timestep)[0],
            self.smart_grid_system.ppm.get_state(timestep),
            self.smart_grid_system.em.get_purchased_units()[timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
        ]))
        return battery_cost + chp_cost + market_cost + prior_purchased_cost + solar_cost + public_grid_cost

    def _calculate_penalties(self):
        penalty = 0
        timestep = min(self.time_step, len(self.smart_grid_system.final_df) - 1)
        total_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep]
        supplied_energy = sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(timestep)[0],
            self.smart_grid_system.ppm.get_state(timestep),
            self.smart_grid_system.em.get_purchased_units()[timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
        ])

        demand_left = total_demand - supplied_energy
        if demand_left > 0:
            penalty += demand_left * 200
        if supplied_energy > total_demand:
            penalty += (supplied_energy - total_demand) * 50

        if demand_left > 0 and supplied_energy < total_demand:
            penalty += 5000

        return penalty

    def _calculate_market_reward(self, market_action_value):
        if market_action_value > 0:
            return market_action_value * 10
        return 0

    def _get_state(self):
        state = self.smart_grid_system.get_state()
        flat_state = []
        flat_state.append(state['time step'])
        flat_state.extend(state['battery'])
        flat_state.extend(state['chp'])
        flat_state.extend(state['electric market'])
        flat_state.extend(state['public grid'])
        flat_state.append(state['prior purchased'])
        flat_state.append(state['solar'])
        flat_state.extend(state['DF'].values())
        flat_state = [float(x) if isinstance(x, (int, float)) else 0 for x in flat_state]
        return np.array(flat_state)

def train_ppo(env, agent1, agent2, num_episodes, max_timesteps):
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        for t in range(max_timesteps):
            battery_action, log_prob1, value1 = agent1.select_action(state)
            solar_action, log_prob2, value2 = agent1.select_action(state)
            chp_action, log_prob3, value3 = agent2.select_action(state)
            market_action, log_prob4, value4 = agent2.select_action(state)
            next_state, reward, done = env.step(battery_action, solar_action, chp_action, market_action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward
            agent1.log_probs.append(log_prob1)
            agent1.values.append(value1)
            agent1.rewards.append(reward)
            agent1.masks.append(1 - done)
            agent1.log_probs.append(log_prob2)
            agent1.values.append(value2)
            agent1.rewards.append(reward)
            agent1.masks.append(1 - done)
            agent2.log_probs.append(log_prob3)
            agent2.values.append(value3)
            agent2.rewards.append(reward)
            agent2.masks.append(1 - done)
            agent2.log_probs.append(log_prob4)
            agent2.values.append(value4)
            agent2.rewards.append(reward)
            agent2.masks.append(1 - done)
            state = next_state
            if done:
                break
        agent1.update()
        agent2.update()
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

# Example usage
datasets = [
    {'name': 'Dataset1', 'sim_file': '../../../data/load_details/simulation_file_20240523_150402.xlsx', 'solar_file': '../../../data/solar_generated_02-02-2024_10_0.15.xlsx'},
    # Add more datasets as needed
]

# Initialize environment and agents
smart_grid_system = SmartGridSystem(datasets[0]['sim_file'], datasets[0]['solar_file'], 0.1)
env = CustomSmartGridEnv(smart_grid_system)
state_dim = len(env._get_state())
discrete_action_dim = 2
continuous_action_dim = 2
lr = 1e-4
gamma = 0.99
epsilon = 0.2
beta = 0.01
num_episodes = 100
max_timesteps = 1441
battery_solar_agent = PPOAgent(DiscretePolicyNetwork(state_dim, discrete_action_dim), lr, gamma, epsilon, beta)
chp_market_agent = PPOAgent(ContinuousPolicyNetwork(state_dim, continuous_action_dim), lr, gamma, epsilon, beta)

# Train the agents
train_ppo(env, battery_solar_agent, chp_market_agent, num_episodes, max_timesteps)

def test_model_on_multiple_datasets(agent1, agent2, datasets, solarcost_kwh):
    results = {}
    for dataset in datasets:
        smart_grid_system = SmartGridSystem(dataset['sim_file'], dataset['solar_file'], solarcost_kwh)
        env = CustomSmartGridEnv(smart_grid_system)
        state = env.reset()
        total_reward = 0
        done = False
        utilization_data = []
        demand_met_data = []
        reward_data = []
        penalty_data = []
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            battery_action, _, _ = agent1.select_action(state)
            solar_action, _, _ = agent1.select_action(state)
            chp_action, _, _ = agent2.select_action(state)
            market_action, _, _ = agent2.select_action(state)
            state, reward, done = env.step(battery_action, solar_action, chp_action, market_action)
            total_reward += reward
            utilization_data.append({
                'battery': env.smart_grid_system.battery.get_state(),
                'chp': env.smart_grid_system.chp.get_state(),
                'electric_market': env.smart_grid_system.em.get_purchased_at_current_step(),
                'public_grid': env.smart_grid_system.pg.get_state(),
                'prior_purchased': env.smart_grid_system.ppm.get_state(env.time_step),
                'solar_output': env.smart_grid_system.solar_model.get_state(env.time_step)
            })
            demand_met_data.append(env.smart_grid_system.final_df['Total_current_demand'].iloc[min(env.time_step, len(env.smart_grid_system.final_df) - 1)])
            reward_data.append(reward)
            penalty_data.append(env._calculate_penalties())
        results[dataset['name']] = {
            'total_reward': total_reward,
            'utilization_data': utilization_data,
            'demand_met_data': demand_met_data,
            'reward_data': reward_data,
            'penalty_data': penalty_data
        }
    return results

def plot_results(results):
    for dataset_name, data in results.items():
        utilization_data = data['utilization_data']
        demand_met_data = data['demand_met_data']
        reward_data = data['reward_data']
        penalty_data = data['penalty_data']

        time_steps = range(len(demand_met_data))

        plt.figure(figsize=(14, 8))

        # Plot Demand Met
        plt.subplot(2, 2, 1)
        plt.plot(time_steps, demand_met_data, label='Demand Met', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Demand Met')
        plt.title(f'Demand Met over Time - {dataset_name}')
        plt.legend()

        # Plot Reward
        plt.subplot(2, 2, 2)
        plt.plot(time_steps, reward_data, label='Reward', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.title(f'Reward over Time - {dataset_name}')
        plt.legend()

        # Plot Penalty
        plt.subplot(2, 2, 3)
        plt.plot(time_steps, penalty_data, label='Penalty', color='red')
        plt.xlabel('Time Step')
        plt.ylabel('Penalty')
        plt.title(f'Penalty over Time - {dataset_name}')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot utilization graphs separately as percentages of total demand
        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        battery_utilization = [(utilization['battery'][0] / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, battery_utilization, label='Battery', color='yellow')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Battery Utilization (%)')
        plt.title(f'Battery Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 2)
        chp_utilization = [(utilization['chp'][0] / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, chp_utilization, label='CHP', color='orange')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('CHP Utilization (%)')
        plt.title(f'CHP Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 3)
        em_utilization = [(utilization['electric_market']['electricity_output'] / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, em_utilization, label='Electric Market', color='cyan')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Electric Market Utilization (%)')
        plt.title(f'Electric Market Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 4)
        pg_utilization = [(utilization['public_grid'][0] / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, pg_utilization, label='Public Grid', color='purple')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Public Grid Utilization (%)')
        plt.title(f'Public Grid Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 5)
        prior_purchased_utilization = [(utilization['prior_purchased'] / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, prior_purchased_utilization, label='Prior Purchased', color='pink')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Prior Purchased Utilization (%)')
        plt.title(f'Prior Purchased Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 6)
        solar_utilization = [(utilization['solar_output'] / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, solar_utilization, label='Solar Output', color='lightgreen')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Solar Output Utilization (%)')
        plt.title(f'Solar Output Utilization over Time - {dataset_name}')
        plt.legend()

        plt.tight_layout()
        plt.show()


def save_model(agent, filepath):
    """
    Save the model parameters to a file.

    Args:
        agent: The PPO agent whose model parameters need to be saved.
        filepath: The file path where the model parameters will be saved.
    """
    torch.save(agent.policy_net.state_dict(), filepath)


def load_model(agent, filepath):
    """
    Load the model parameters from a file.

    Args:
        agent: The PPO agent whose model parameters need to be loaded.
        filepath: The file path from where the model parameters will be loaded.
    """
    agent.policy_net.load_state_dict(torch.load(filepath))
    agent.policy_net.eval()  # Set the model to evaluation mode


# Save the model
save_model(battery_solar_agent, 'battery_solar_agent.pth')
save_model(chp_market_agent, 'chp_market_agent.pth')

# Load the model
load_model(battery_solar_agent, 'battery_solar_agent.pth')
load_model(chp_market_agent, 'chp_market_agent.pth')

# Test the model on multiple datasets
results = test_model_on_multiple_datasets(battery_solar_agent, chp_market_agent, datasets, solarcost_kwh=0.1)

# Plot the results
plot_results(results)
