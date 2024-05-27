from methods.Torch_ppo.custom_env import CustomSmartGridEnv
from methods.Torch_ppo.ppo_agent import PPOAgent
from models.smart_grid_system import SmartGridSystem
import torch

def train_ppo(env, agent, num_episodes, max_timesteps):
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)  # Ensure state is a float tensor
        for t in range(max_timesteps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Ensure next_state is a float tensor

            agent.log_probs.append(log_prob)
            agent.values.append(value)
            agent.rewards.append(reward)
            agent.masks.append(1 - done)

            state = next_state
            if done:
                break

        agent.update()
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {sum(agent.rewards)}")

# Initialize environment and agent
smart_grid_system = SmartGridSystem('../../../data/load_details/simulation_file_20240523_150402.xlsx',
                                    '../../../data/solar_generated_02-02-2024_10_0.15.xlsx', 0.1)

env = CustomSmartGridEnv(smart_grid_system)

state_dim = env.observation_space.shape[0]
discrete_action_dim = 2  # Example: battery and solar actions
continuous_action_dim = 2  # Example: CHP and market purchase actions

lr = 1e-4
gamma = 0.99
epsilon = 0.2
beta = 0.01
num_episodes = 1000
max_timesteps = 1000

agent = PPOAgent(state_dim, discrete_action_dim, continuous_action_dim, lr, gamma, epsilon, beta)

# Train the agent
train_ppo(env, agent, num_episodes, max_timesteps)
