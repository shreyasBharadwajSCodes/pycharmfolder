import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import matplotlib.pyplot as plt

from models.smart_grid_system import SmartGridSystem

# Set up logging
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
        masks = torch.FloatTensor(self.masks).squeeze()
        returns = rewards
        advantage = returns - values

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

# CustomSmartGridEnv Class
class CustomSmartGridEnv:
    def __init__(self, smart_grid_system):
        self.smart_grid_system = smart_grid_system
        self.time_step = 0
        self.previous_demand = 0  # Initialize previous demand

    def reset(self):
        self.smart_grid_system.reset()
        self.time_step = 0
        self.previous_demand = 0  # Reset previous demand
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
        reward += self._calculate_solar_reward()  # Additional reward for using solar energy

        logging.info(
            f"TimeStep: {self.time_step}, Battery Action: {battery_action}, Solar Action: {solar_action}, "
            f"CHP Action: {chp_action_value}, Market Action: {market_action_value}, Costs: {costs}, Penalties: {penalties}, Reward: {reward}")

        # Update previous demand
        self.previous_demand = self.smart_grid_system.final_df['Total_current_demand'].iloc[min(self.time_step, len(self.smart_grid_system.final_df) - 1)]

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
                self.smart_grid_system.battery.discharge(demand_left)
            else:
                self.smart_grid_system.battery.discharge(0)  # to avoid excess discharge

    def _apply_solar_action(self, action):
        if action == 0:
            self.smart_grid_system.solar_model.set_mode("off")
        elif action == 1:
            self.smart_grid_system.solar_model.set_mode("on")
        logging.info(f"Solar Action: {action}")

    def _apply_chp_action(self, action):
        self.smart_grid_system.chp.set_output(action)
        logging.info(f"CHP Action: {action}")

    def _apply_market_purchase_action(self, action):
        if 6 <= self.time_step % 15 <= 9:
            units = max(0, action)
            self.smart_grid_system.em.make_purchase(units)
            logging.info(f"Market Purchase Action: {units}")

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
                timestep // 15 if timestep // 15 < len(self.smart_grid_system.em.get_purchased_units()) else len(self.smart_grid_system.em.get_purchased_units()) - 1]
        ])

        if public_grid_usage < 0.02 * self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep]:
            public_grid_cost = 0.01 * public_grid_usage  # very low cost
        else:
            public_grid_cost = self.smart_grid_system.pg.get_price(self.smart_grid_system.final_df['Total_current_demand'].iloc[timestep], public_grid_usage)

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
            penalty += demand_left * 100  # reduced penalty
        if supplied_energy > total_demand:
            penalty += (supplied_energy - total_demand) * 25  # reduced penalty

        if demand_left > 0 and supplied_energy < total_demand:
            penalty += 1000  # reduced penalty

        # Adjust penalty for public grid
        public_grid_usage = total_demand - supplied_energy
        if public_grid_usage > 0.02 * total_demand:
            penalty += 500  # larger penalty if above threshold

        logging.info(f"Total Penalties: {penalty}")
        return penalty

    def _calculate_solar_reward(self):
        solar_output = self.smart_grid_system.solar_model.get_state(self.time_step)[0]
        return solar_output * 10  # specific reward for solar usage

    def _calculate_market_reward(self, market_action_value):
        market_reward = market_action_value * 10 if market_action_value > 0 else 0
        logging.info(f"Market Reward: {market_reward}")
        return market_reward

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
        flat_state.append(self.smart_grid_system.final_df['Total_current_demand'].iloc[
            min(self.time_step, len(self.smart_grid_system.final_df) - 1)])  # Add current total demand
        flat_state = [float(x) if isinstance(x, (int, float)) else 0 for x in flat_state]
        return np.array(flat_state)

def train_ppo(env, battery_agent, market_agent, num_episodes, max_timesteps):
    battery_action_counts = {0: 0, 1: 0, 2: 0}
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        episode_reward = 0
        for t in range(max_timesteps):
            battery_action, log_prob1, value1 = battery_agent.select_action(state)
            solar_action = 1  # Always on unless demand is too low
            chp_action, log_prob2, value2 = battery_agent.select_action(state)
            market_action, log_prob3, value3 = market_agent.select_action(state)
            next_state, reward, done = env.step(battery_action, solar_action, chp_action, market_action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            episode_reward += reward
            battery_agent.log_probs.append(log_prob1)
            battery_agent.values.append(value1)
            battery_agent.rewards.append(reward)
            battery_agent.masks.append(1 - done)
            battery_agent.log_probs.append(log_prob2)
            battery_agent.values.append(value2)
            battery_agent.rewards.append(reward)
            battery_agent.masks.append(1 - done)
            market_agent.log_probs.append(log_prob3)
            market_agent.values.append(value3)
            market_agent.rewards.append(reward)
            market_agent.masks.append(1 - done)
            state = next_state
            battery_action_counts[battery_action] += 1
            if done:
                break
        battery_agent.update()
        market_agent.update()
        if episode % 10 == 0:
            logging.info(f"Episode {episode}, Reward: {episode_reward}")
            logging.info(f"Battery Action Distribution: {battery_action_counts}")

def test_model_on_multiple_datasets(battery_agent, market_agent, datasets, solarcost_kwh):
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
            battery_action, _, _ = battery_agent.select_action(state)
            solar_action = 1  # Always on unless demand is too low
            chp_action, _, _ = battery_agent.select_action(state)
            market_action, _, _ = market_agent.select_action(state)
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
            demand_met_data.append(env.smart_grid_system.final_df['Total_current_demand'].iloc[
                min(env.time_step, len(env.smart_grid_system.final_df) - 1)])
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

def save_model(agent, filepath):
    torch.save(agent.policy_net.state_dict(), filepath)

def load_model(agent, filepath):
    agent.policy_net.load_state_dict(torch.load(filepath))
    agent.policy_net.eval()

# Initialize environment and agents
datasets = [
    {'name': 'Dataset1', 'sim_file': '../../data/load_details/simulation_file_20240523_150402.xlsx',
     'solar_file': '../../data/solar_generated_02-02-2024_10_0.15.xlsx'},
    # Add more datasets as needed
]

smart_grid_system = SmartGridSystem(datasets[0]['sim_file'], datasets[0]['solar_file'], 0.1)
env = CustomSmartGridEnv(smart_grid_system)
state_dim = len(env._get_state())
battery_action_dim = 3  # Three actions: idle, charge, discharge
market_action_dim = 10  # Ten discrete actions for market purchases
lr = 1e-4
gamma = 0.90
epsilon = 0.2
beta = 0.02
num_episodes = 10
max_timesteps = 1441

battery_agent = PPOAgent(DiscretePolicyNetwork(state_dim, battery_action_dim), lr, gamma, epsilon, beta)
market_agent = PPOAgent(DiscretePolicyNetwork(state_dim, market_action_dim), lr, gamma, epsilon, beta)

# Train the agents
start_time = time.time()
train_ppo(env, battery_agent, market_agent, num_episodes, max_timesteps)

# Save the models
save_model(battery_agent, 'battery_agent.pth')
save_model(market_agent, 'market_agent.pth')

# Test the model on multiple datasets
results = test_model_on_multiple_datasets(battery_agent, market_agent, datasets, solarcost_kwh=0.1)

end_time = time.time()
logging.info(f"Finished execution in {end_time - start_time} seconds")
print(f"Finished execution in {end_time - start_time} seconds")

def visualize_results(results):
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
            utilization['electric_market']['electricity_output'] if isinstance(utilization['electric_market'],
                                                                               dict) else utilization[
                'electric_market'],
            utilization['prior_purchased'] if isinstance(utilization['prior_purchased'], (int, float)) else 0,
            utilization['solar_output'] if isinstance(utilization['solar_output'], (int, float)) else 0
        ]) / demand_met_data[i]) * 100 for i, utilization in enumerate(utilization_data)]
        plt.subplot(2, 2, 1)
        plt.plot(time_steps, percentage_demand_satisfied, label='% Demand Satisfied by Non-Public Grid Sources',
                 color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('% Demand Satisfied')
        plt.title(f'% Demand Satisfied by Non-Public Grid Sources - {dataset_name}')
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
        battery_utilization = [
            (utilization['battery'][0] / demand_met_data[i]) * 100 if isinstance(utilization['battery'], list) else (
                                                                                                                                utilization[
                                                                                                                                    'battery'] /
                                                                                                                                demand_met_data[
                                                                                                                                    i]) * 100
            for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, battery_utilization, label='Battery', color='yellow')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Battery Utilization (%)')
        plt.title(f'Battery Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 2)
        chp_utilization = [
            (utilization['chp'][0] / demand_met_data[i]) * 100 if isinstance(utilization['chp'], list) else (
                                                                                                                        utilization[
                                                                                                                            'chp'] /
                                                                                                                        demand_met_data[
                                                                                                                            i]) * 100
            for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, chp_utilization, label='CHP', color='orange')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('CHP Utilization (%)')
        plt.title(f'CHP Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 3)
        em_utilization = [
            (utilization['electric_market']['electricity_output'] / demand_met_data[i]) * 100 if isinstance(
                utilization['electric_market'], dict) else (utilization['electric_market'] / demand_met_data[i]) * 100
            for i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, em_utilization, label='Electric Market', color='cyan')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Electric Market Utilization (%)')
        plt.title(f'Electric Market Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 4)
        prior_purchased_utilization = [
            (utilization['prior_purchased'] / demand_met_data[i]) * 100 if isinstance(utilization['prior_purchased'],
                                                                                      (int, float)) else 0 for
            i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, prior_purchased_utilization, label='Prior Purchased', color='pink')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Prior Purchased Utilization (%)')
        plt.title(f'Prior Purchased Utilization over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 5)
        solar_utilization = [
            (utilization['solar_output'] / demand_met_data[i]) * 100 if isinstance(utilization['solar_output'],
                                                                                   (int, float)) else 0 for
            i, utilization in enumerate(utilization_data)]
        plt.plot(time_steps, solar_utilization, label='Solar Output', color='lightgreen')
        plt.ylim(0, 100)
        plt.xlabel('Time Step')
        plt.ylabel('Solar Output Utilization (%)')
        plt.title(f'Solar Output Utilization over Time - {dataset_name}')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot total units of energy used by each source
        plt.figure(figsize=(14, 10))

        plt.subplot(3, 2, 1)
        battery_units = [
            utilization['battery'][0] if isinstance(utilization['battery'], list) else utilization['battery'] for
            utilization in utilization_data]
        plt.plot(time_steps, battery_units, label='Battery Units', color='yellow')
        plt.xlabel('Time Step')
        plt.ylabel('Battery Units')
        plt.title(f'Battery Units Consumed over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 2)
        chp_units = [utilization['chp'][0] if isinstance(utilization['chp'], list) else utilization['chp'] for
                     utilization in utilization_data]
        plt.plot(time_steps, chp_units, label='CHP Units', color='orange')
        plt.xlabel('Time Step')
        plt.ylabel('CHP Units')
        plt.title(f'CHP Units Consumed over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 3)
        em_units = [utilization['electric_market']['electricity_output'] if isinstance(utilization['electric_market'],
                                                                                       dict) else utilization[
            'electric_market'] for utilization in utilization_data]
        plt.plot(time_steps, em_units, label='Electric Market Units', color='cyan')
        plt.xlabel('Time Step')
        plt.ylabel('Electric Market Units')
        plt.title(f'Electric Market Units Consumed over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 4)
        prior_purchased_units = [
            utilization['prior_purchased'] if isinstance(utilization['prior_purchased'], (int, float)) else 0 for
            utilization in utilization_data]
        plt.plot(time_steps, prior_purchased_units, label='Prior Purchased Units', color='pink')
        plt.xlabel('Time Step')
        plt.ylabel('Prior Purchased Units')
        plt.title(f'Prior Purchased Units Consumed over Time - {dataset_name}')
        plt.legend()

        plt.subplot(3, 2, 5)
        solar_units = [utilization['solar_output'] if isinstance(utilization['solar_output'], (int, float)) else 0 for
                       utilization in utilization_data]
        plt.plot(time_steps, solar_units, label='Solar Output Units', color='lightgreen')
        plt.xlabel('Time Step')
        plt.ylabel('Solar Output Units')
        plt.title(f'Solar Output Units Consumed over Time - {dataset_name}')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot total demand units
        plt.figure(figsize=(14, 5))
        plt.plot(time_steps, demand_met_data, label='Total Demand Units', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Total Demand Units')
        plt.title(f'Total Demand Units over Time - {dataset_name}')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Visualize the results
visualize_results(results)
