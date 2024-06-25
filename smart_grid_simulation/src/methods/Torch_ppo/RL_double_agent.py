import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from models.smart_grid_system import SmartGridSystem  # Ensure this module is available

# Initialize logging
logging.basicConfig(filename='Rl_smart_grid.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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
        action_probs, value = self.policy_net(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob, value

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

# DiscretePolicyNetwork Class with Increased Complexity
class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# CustomSmartGridEnv Class with Updated Reward Function
class CustomSmartGridEnv:
    def __init__(self, smart_grid_system):
        self.smart_grid_system = smart_grid_system
        self.time_step = 0
        self.max_purchase = 100
        self.previous_demand = 0  # Initialize previous demand

    def reset(self):
        self.smart_grid_system.reset()
        self.time_step = 0
        self.previous_demand = 0  # Reset previous demand
        return self._get_state()

    def step(self, battery_action, chp_action, market_action):
        battery_action = int(battery_action)
        chp_action = int(chp_action)
        market_action = int(market_action)

        self._apply_battery_action(battery_action)
        self._apply_solar_action()  # Solar action is always on or off
        self._apply_chp_action(chp_action)
        self._apply_market_purchase_action(market_action)

        costs = self._calculate_costs()
        penalties = self._calculate_penalties()
        reward = -(costs + penalties)

        # Apply exponential penalty for public grid usage above threshold
        public_grid_usage = self.smart_grid_system.final_df['Total_current_demand'].iloc[self.time_step] - sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(self.time_step)[0],
            self.smart_grid_system.ppm.get_state(self.time_step),
            self.smart_grid_system.em.get_purchased_units()[self.time_step // 15 if self.time_step // 15 < len(
                self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
        ])
        if public_grid_usage < 0:
            public_grid_usage = 0  # Ensure no negative usage

        total_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[self.time_step]
        if public_grid_usage / total_demand > 0.02:
            penalty = (public_grid_usage / total_demand - 0.02) * total_demand * 500  # Strong penalty
            reward -= penalty

        # Separate penalty for unmet demand
        demand_left = total_demand - sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(self.time_step)[0],
            self.smart_grid_system.ppm.get_state(self.time_step),
            self.smart_grid_system.em.get_purchased_units()[self.time_step // 15 if self.time_step // 15 < len(
                self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
        ])
        if demand_left > 0:
            unmet_demand_penalty = demand_left * 500  # Separate penalty for unmet demand
            reward -= unmet_demand_penalty

        # Update previous demand
        self.previous_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[
            min(self.time_step, len(self.smart_grid_system.final_df) - 1)]

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
                self.smart_grid_system.em.get_purchased_units()[self.time_step // 15 if self.time_step // 15 < len(
                    self.smart_grid_system.em.get_purchased_units()) else len(
                    self.smart_grid_system.em.get_purchased_units()) - 1]
            ])
            self.smart_grid_system.battery.charge(available_energy)
        elif action == 2:
            self.smart_grid_system.battery.set_mode("discharge")
            total_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[
                min(self.time_step, len(self.smart_grid_system.final_df) - 1)]
            supplied_energy = sum([
                self.smart_grid_system.chp.current_output,
                self.smart_grid_system.solar_model.get_state(self.time_step)[0],
                self.smart_grid_system.ppm.get_state(self.time_step),
                self.smart_grid_system.em.get_purchased_units()[self.time_step // 15 if self.time_step // 15 < len(
                    self.smart_grid_system.em.get_purchased_units()) else len(
                    self.smart_grid_system.em.get_purchased_units()) - 1]
            ])
            demand_left = total_demand - supplied_energy
            if demand_left > 0:
                self.smart_grid_system.battery.discharge()
            else:
                self.smart_grid_system.battery.discharge(0)  # to avoid excess discharge

    def _apply_solar_action(self):
        solar_output = self.smart_grid_system.final_df['Solar kw/min'].iloc[
            min(self.time_step, len(self.smart_grid_system.final_df) - 1)]
        logging.info(f"Solar Output: {solar_output}")

    def _apply_chp_action(self, action):
        if action == 0:
            self.smart_grid_system.chp.set_output(0)
        elif action == 1:
            self.smart_grid_system.chp.set_output(self.smart_grid_system.chp.min_operational_value)
        elif action == 2:
            self.smart_grid_system.chp.set_output(self.smart_grid_system.chp.max_operational_value)
        logging.info(f"CHP Action: {action}")

    def _apply_market_purchase_action(self, action):
        market_action_value = self._get_market_action_value(action)
        if 6 <= self.time_step % 15 <= 9:
            units = max(0, market_action_value)
            self.smart_grid_system.em.make_purchase(units)
            logging.info(f"Market Purchase Action: {units}")

    def _get_market_action_value(self, action):
        return action * (self.max_purchase / 10)  # Discretize into 10 actions

    def _calculate_costs(self):
        timestep = min(self.time_step, len(self.smart_grid_system.final_df) - 1)
        battery_cost = self.smart_grid_system.battery.get_cost()
        chp_cost = self.smart_grid_system.chp.calculate_cost_at_current_step()
        market_cost = self.smart_grid_system.em.get_price(timestep) * self.smart_grid_system.em.get_purchased_units()[
            timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(
                self.smart_grid_system.em.get_purchased_units()) - 1]
        prior_purchased_cost = self.smart_grid_system.ppm.get_price(timestep)
        solar_cost = self.smart_grid_system.solar_model.calculate_cost_at_timestep(timestep)

        # Public Grid Cost adjustment
        public_grid_usage = self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep] - sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(timestep)[0],
            self.smart_grid_system.ppm.get_state(timestep),
            self.smart_grid_system.em.get_purchased_units()[
                timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(
                    self.smart_grid_system.em.get_purchased_units()) - 1]
        ])

        if public_grid_usage < 0:
            public_grid_usage = 0  # Ensure no negative usage

        if public_grid_usage < 0.02 * self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep]:
            public_grid_cost = 0.01 * public_grid_usage  # very low cost
        else:
            public_grid_cost = self.smart_grid_system.pg.get_price(
                self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep], public_grid_usage)

        total_cost = battery_cost + chp_cost + market_cost + prior_purchased_cost + solar_cost + public_grid_cost
        logging.info(f"Total Costs: {total_cost}")
        return total_cost

    def _calculate_penalties(self):
        penalty = 0
        timestep = min(self.time_step, len(self.smart_grid_system.final_df) - 1)
        total_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep]
        supplied_energy = sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(timestep)[0],
            self.smart_grid_system.ppm.get_state(timestep),
            self.smart_grid_system.em.get_purchased_units()[
                timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(
                    self.smart_grid_system.em.get_purchased_units()) - 1]
        ])

        demand_left = total_demand - supplied_energy
        if demand_left > 0:
            penalty += demand_left * 500  # Separate penalty for unmet demand
        if supplied_energy > total_demand:
            penalty += (supplied_energy - total_demand) * 25  # Penalty for excess supply

        if demand_left > 0 and supplied_energy < total_demand:
            penalty += 1000  # Additional penalty for unmet demand

        logging.info(f"Total Penalties: {penalty}")
        return penalty

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
        flat_state.append(self.time_step)  # Add current timestep to the state
        flat_state.append(self.previous_demand)  # Add previous demand to the state
        flat_state = [float(x) if isinstance(x, (int, float)) else 0 for x in flat_state]
        return np.array(flat_state)

# Define the main training and testing functions
def train_ppo(env, market_agent, battery_agent, num_episodes, max_timesteps):
    start_time = time.time()
    battery_action_counts = {0: 0, 1: 0, 2: 0}  # Track battery action distribution
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        for t in range(max_timesteps):
            market_action, log_prob1, value1 = market_agent.select_action(state)
            battery_action, log_prob2, value2 = battery_agent.select_action(state)
            chp_action = battery_action  # Assuming the same action for simplicity; modify as needed

            next_state, reward, done = env.step(battery_action, chp_action, market_action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward
            market_agent.log_probs.append(log_prob1)
            market_agent.values.append(value1)
            market_agent.rewards.append(reward)
            market_agent.masks.append(1 - done)
            battery_agent.log_probs.append(log_prob2)
            battery_agent.values.append(value2)
            battery_agent.rewards.append(reward)
            battery_agent.masks.append(1 - done)
            state = next_state
            battery_action_counts[battery_action] += 1  # Count battery actions
            if done:
                break
        market_agent.update()
        battery_agent.update()
        if episode % 10 == 0:
            logging.info(f"Episode {episode}, Reward: {episode_reward}")
            # Log the actions chosen by the agent
            logging.info(f"Battery Action Distribution: {battery_action_counts}")
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Finished execution in {execution_time} seconds.")

datasets = [
    {'name': 'Dataset1', 'sim_file': '../../../data/load_details/simulation_file_20240523_150402.xlsx',
     'solar_file': '../../../data/solar_generated_02-02-2024_10_0.15.xlsx'},
    # Add more datasets as needed
]

# Initialize environment and agents
smart_grid_system = SmartGridSystem(datasets[0]['sim_file'], datasets[0]['solar_file'], 0.1)
env = CustomSmartGridEnv(smart_grid_system)
state_dim = len(env._get_state())
battery_action_dim = 3  # Three actions: idle, charge, discharge
chp_action_dim = 3  # Three actions: off, min, max
market_action_dim = 10  # Discretized into 10 actions
lr = 1e-4
gamma = 0.90
epsilon = 0.2
beta = 0.02
num_episodes = 40  # Increased number of episodes
max_timesteps = 1441
market_agent = PPOAgent(DiscretePolicyNetwork(state_dim, market_action_dim), lr, gamma, epsilon, beta)
battery_agent = PPOAgent(DiscretePolicyNetwork(state_dim, battery_action_dim), lr, gamma, epsilon, beta)

# Train the agents
train_ppo(env, market_agent, battery_agent, num_episodes, max_timesteps)

# Define test function
def test_model_on_multiple_datasets(market_agent, battery_agent, datasets, solarcost_kwh):
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
        cost_data = []  # Added for cost tracking
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            market_action, _, _ = market_agent.select_action(state)
            battery_action, _, _ = battery_agent.select_action(state)
            chp_action = battery_action  # Assuming the same action for simplicity; modify as needed
            state, reward, done = env.step(battery_action, chp_action, market_action)
            total_reward += reward
            utilization_data.append({
                'battery': env.smart_grid_system.battery.get_state(),
                'chp': env.smart_grid_system.chp.get_state(),
                'electric_market': env.smart_grid_system.em.get_purchased_at_current_step(),
                'public_grid': env.smart_grid_system.pg.get_state(),
                'prior_purchased': env.smart_grid_system.ppm.get_state(env.time_step),
                'solar_output': env.smart_grid_system.final_df['Solar kw/min'].iloc[min(env.time_step, len(env.smart_grid_system.final_df) - 1)]
            })
            demand_met_data.append(env.smart_grid_system.final_df['Total_current_demand'].iloc[
                                       min(env.time_step, len(env.smart_grid_system.final_df) - 1)])
            reward_data.append(reward)
            penalty_data.append(env._calculate_penalties())
            cost_data.append(env._calculate_costs())  # Add cost for each timestep
        results[dataset['name']] = {
            'total_reward': total_reward,
            'utilization_data': utilization_data,
            'demand_met_data': demand_met_data,
            'reward_data': reward_data,
            'penalty_data': penalty_data,
            'cost_data': cost_data  # Include cost data in results
        }
    return results

# Save and load model functions
def save_model(agent, filepath):
    torch.save(agent.policy_net.state_dict(), filepath)

def load_model(agent, filepath):
    agent.policy_net.load_state_dict(torch.load(filepath))
    agent.policy_net.eval()  # Set the model to evaluation mode

# Save the models
save_model(market_agent, 'market_agent.pth')
save_model(battery_agent, 'battery_agent.pth')

# Load the models
load_model(market_agent, 'market_agent.pth')
load_model(battery_agent, 'battery_agent.pth')

# Test the model on multiple datasets
results = test_model_on_multiple_datasets(market_agent, battery_agent, datasets, solarcost_kwh=0.1)

def plot_results(results):
    for dataset_name, data in results.items():
        utilization_data = data['utilization_data']
        demand_met_data = data['demand_met_data']
        reward_data = data['reward_data']
        penalty_data = data['penalty_data']

        time_steps = range(len(demand_met_data))

        plt.figure(figsize=(14, 8))

        # Plot Percentage of Demand Satisfied by Sources Other Than Public Grid
        percentage_demand_satisfied = [(sum([
            utilization['battery'][0] if isinstance(utilization['battery'], list) else utilization['battery'],
            utilization['chp'][0] if isinstance(utilization['chp'], list) else utilization['chp'],
            utilization['electric_market']['electricity_output'] if isinstance(utilization['electric_market'], dict) else utilization['electric_market'],
            utilization['prior_purchased'] if isinstance(utilization['prior_purchased'], (int, float)) else 0,
            utilization['solar_output'] if isinstance(utilization['solar_output'], (int, float)) else 0
        ]) / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.subplot(2, 2, 1)
        plt.plot(time_steps, percentage_demand_satisfied, label='% Demand Satisfied by Non-Public Grid Sources', color='blue', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('% Demand Satisfied', fontsize=12)
        plt.title(f'% Demand Satisfied by Non-Public Grid Sources - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        # Plot Reward
        plt.subplot(2, 2, 2)
        plt.plot(time_steps, reward_data, label='Reward', color='green', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title(f'Reward over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        # Plot Penalty
        plt.subplot(2, 2, 3)
        plt.plot(time_steps, penalty_data, label='Penalty', color='red', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Penalty', fontsize=12)
        plt.title(f'Penalty over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

        # Plot utilization graphs separately as percentages of total demand
        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        battery_utilization = [
            (utilization['battery'][0] / demand_met_data[i]) * 100 if isinstance(utilization['battery'], list) else (
                                                                                                                                utilization[
                                                                                                                                    'battery'] /
                                                                                                                                demand_met_data[
                                                                                                                                    i]) * 100
            for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, battery_utilization, label='Battery', color='yellow', linewidth=2)
        plt.ylim(0, 100)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Battery Utilization (%)', fontsize=12)
        plt.title(f'Battery Utilization over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 2)
        chp_utilization = [
            (utilization['chp'][0] / demand_met_data[i]) * 100 if isinstance(utilization['chp'], list) else (
                                                                                                                        utilization[
                                                                                                                            'chp'] /
                                                                                                                        demand_met_data[
                                                                                                                            i]) * 100
            for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, chp_utilization, label='CHP', color='orange', linewidth=2)
        plt.ylim(0, 100)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('CHP Utilization (%)', fontsize=12)
        plt.title(f'CHP Utilization over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 3)
        em_utilization = [
            (utilization['electric_market']['electricity_output'] / demand_met_data[i]) * 100 if isinstance(
                utilization['electric_market'], dict) else (utilization['electric_market'] / demand_met_data[i]) * 100
            for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, em_utilization, label='Electric Market', color='cyan', linewidth=2)
        plt.ylim(0, 100)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Electric Market Utilization (%)', fontsize=12)
        plt.title(f'Electric Market Utilization over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 4)
        pg_utilization = [
            (utilization['public_grid'][0] / demand_met_data[i]) * 100 if isinstance(utilization['public_grid'],
                                                                                     list) else (utilization[
                                                                                                     'public_grid'] /
                                                                                                 demand_met_data[
                                                                                                     i]) * 100 for
            i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, pg_utilization, label='Public Grid', color='purple', linewidth=2)
        plt.ylim(0, 100)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Public Grid Utilization (%)', fontsize=12)
        plt.title(f'Public Grid Utilization over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 5)
        prior_purchased_utilization = [
            (utilization['prior_purchased'] / demand_met_data[i]) * 100 if isinstance(utilization['prior_purchased'],
                                                                                      (int, float)) else 0 for
            i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, prior_purchased_utilization, label='Prior Purchased', color='pink', linewidth=2)
        plt.ylim(0, 100)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Prior Purchased Utilization (%)', fontsize=12)
        plt.title(f'Prior Purchased Utilization over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 6)
        solar_utilization = [
            (utilization['solar_output'] / demand_met_data[i]) * 100 if isinstance(utilization['solar_output'],
                                                                                   (int, float)) else 0 for
            i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, solar_utilization, label='Solar Output', color='lightgreen', linewidth=2)
        plt.ylim(0, 100)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Solar Output Utilization (%)', fontsize=12)
        plt.title(f'Solar Output Utilization over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

        # Plot total units of energy used by each source
        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        battery_units = [
            utilization['battery'][0] if isinstance(utilization['battery'], list) else utilization['battery'] for
            utilization in utilization_data]
        plt.plot(time_steps, battery_units, label='Battery Units', color='yellow', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Battery Units', fontsize=12)
        plt.title(f'Battery Units Consumed over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 2)
        chp_units = [utilization['chp'][0] if isinstance(utilization['chp'], list) else utilization['chp'] for
                     utilization in utilization_data]
        plt.plot(time_steps, chp_units, label='CHP Units', color='orange', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('CHP Units', fontsize=12)
        plt.title(f'CHP Units Consumed over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 3)
        em_units = [utilization['electric_market']['electricity_output'] if isinstance(utilization['electric_market'],
                                                                                       dict) else utilization[
            'electric_market'] for utilization in utilization_data]
        plt.plot(time_steps, em_units, label='Electric Market Units', color='cyan', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Electric Market Units', fontsize=12)
        plt.title(f'Electric Market Units Consumed over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 4)
        pg_units = [utilization['public_grid'][0] if isinstance(utilization['public_grid'], list) else utilization[
            'public_grid'] for utilization in utilization_data]
        plt.plot(time_steps, pg_units, label='Public Grid Units', color='purple', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Public Grid Units', fontsize=12)
        plt.title(f'Public Grid Units Consumed over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 5)
        prior_purchased_units = [
            utilization['prior_purchased'] if isinstance(utilization['prior_purchased'], (int, float)) else 0 for
            utilization in utilization_data]
        plt.plot(time_steps, prior_purchased_units, label='Prior Purchased Units', color='pink', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Prior Purchased Units', fontsize=12)
        plt.title(f'Prior Purchased Units Consumed over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.subplot(3, 2, 6)
        solar_units = [utilization['solar_output'] if isinstance(utilization['solar_output'], (int, float)) else 0 for
                       utilization in utilization_data]
        plt.plot(time_steps, solar_units, label='Solar Output Units', color='lightgreen', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Solar Output Units', fontsize=12)
        plt.title(f'Solar Output Units Consumed over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

        # Plot total demand units
        plt.figure(figsize=(14, 5))
        plt.plot(time_steps, demand_met_data, label='Total Demand Units', color='blue', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Total Demand Units', fontsize=12)
        plt.title(f'Total Demand Units over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        # Calculate running total prices up to each timestep
        running_total_prices = [0] * len(time_steps)
        running_battery_prices = [0] * len(time_steps)
        running_chp_prices = [0] * len(time_steps)
        running_em_prices = [0] * len(time_steps)
        running_prior_purchased_prices = [0] * len(time_steps)
        running_solar_prices = [0] * len(time_steps)

        for i in range(len(time_steps)):
            if i == 0:
                running_battery_prices[i] = utilization_data[i]['battery'][0] if isinstance(utilization_data[i]['battery'], list) else utilization_data[i]['battery']
                running_chp_prices[i] = utilization_data[i]['chp'][0] if isinstance(utilization_data[i]['chp'], list) else utilization_data[i]['chp']
                running_em_prices[i] = utilization_data[i]['electric_market']['electricity_output'] if isinstance(utilization_data[i]['electric_market'], dict) else utilization_data[i]['electric_market']
                running_prior_purchased_prices[i] = utilization_data[i]['prior_purchased'] if isinstance(utilization_data[i]['prior_purchased'], (int, float)) else 0
                running_solar_prices[i] = utilization_data[i]['solar_output'] if isinstance(utilization_data[i]['solar_output'], (int, float)) else 0
            else:
                running_battery_prices[i] = running_battery_prices[i-1] + (utilization_data[i]['battery'][0] if isinstance(utilization_data[i]['battery'], list) else utilization_data[i]['battery'])
                running_chp_prices[i] = running_chp_prices[i-1] + (utilization_data[i]['chp'][0] if isinstance(utilization_data[i]['chp'], list) else utilization_data[i]['chp'])
                running_em_prices[i] = running_em_prices[i-1] + (utilization_data[i]['electric_market']['electricity_output'] if isinstance(utilization_data[i]['electric_market'], dict) else utilization_data[i]['electric_market'])
                running_prior_purchased_prices[i] = running_prior_purchased_prices[i-1] + (utilization_data[i]['prior_purchased'] if isinstance(utilization_data[i]['prior_purchased'], (int, float)) else 0)
                running_solar_prices[i] = running_solar_prices[i-1] + (utilization_data[i]['solar_output'] if isinstance(utilization_data[i]['solar_output'], (int, float)) else 0)

            running_total_prices[i] = running_battery_prices[i] + running_chp_prices[i] + running_em_prices[i] + running_prior_purchased_prices[i] + running_solar_prices[i]

        # Store the running total prices into an Excel file
        df_prices = pd.DataFrame({
            'Time Step': time_steps,
            'Running Battery Prices': running_battery_prices,
            'Running CHP Prices': running_chp_prices,
            'Running Electric Market Prices': running_em_prices,
            'Running Prior Purchased Prices': running_prior_purchased_prices,
            'Running Solar Prices': running_solar_prices,
            'Running Total Prices': running_total_prices
        })
        df_prices.to_excel(f'{dataset_name}_running_prices.xlsx', index=False)

        # Plot running total prices over time
        plt.figure(figsize=(14, 8))
        plt.plot(time_steps, running_total_prices, label='Running Total Prices', color='black', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Running Total Prices', fontsize=12)
        plt.title(f'Running Total Prices over Time - {dataset_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
# Plot the results
plot_results(results)
