import gym
from gym import spaces
import numpy as np
from models.smart_grid_system import SmartGridSystem


class SmartGridEnv(gym.Env):
    def __init__(self, sim_file, solar_file, solarcost):
        super(SmartGridEnv, self).__init__()
        self.solarcost = solarcost
        self.smart_grid = SmartGridSystem(sim_file, solar_file, solarcost)
        self.action_space = spaces.Box(low=0, high=1, shape=(6,),
                                       dtype=np.float32)  # Define action space as a vector of proportions

        # Calculate the size of the observation space
        observation_example = self._get_observation()
        self.observation_space_size = len(observation_example)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.observation_space_size,), dtype=np.float32)

    def reset(self):
        self.smart_grid.time_step = 0
        self.smart_grid.em.reset()
        return self._get_observation()

    def step(self, action):
        self.smart_grid.time_step += 1
        if self.smart_grid.time_step >= len(self.smart_grid.final_df):
            # If time_step exceeds DataFrame length, end the episode
            done = True
            observation = self._get_observation()
            return observation, 0, done, {}  # Return a high penalty if out-of-bounds
        reward, penalty = self._calculate_reward(action)
        done = self.smart_grid.time_step >= len(self.smart_grid.final_df)
        return self._get_observation(), reward - penalty, done, {}

    def _get_observation(self):
        if self.smart_grid.time_step >= len(self.smart_grid.final_df):
            # If time_step exceeds DataFrame length, return default observation
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        state = self.smart_grid.get_state()
        observation = []
        observation.extend(state['battery'])  # List of scalar values
        observation.extend(state['chp'])  # List of scalar values (4 values)
        observation.extend(state['electric market'])  # List of scalar values
        observation.extend(state['public grid'])  # List of scalar values
        observation.append(state['prior purchased'])  # Single scalar value
        observation.append(state['solar'])  # Single scalar value
        observation.append(state['time step'])  # Single scalar value
        observation.append(state['DF']['Total_previous_demand'])  # Single scalar value

        # Ensure all elements are float
        observation = [float(x) for x in observation]
        return np.array(observation, dtype=np.float32)

    def _calculate_reward(self, action):
        if self.smart_grid.time_step >= len(self.smart_grid.final_df):
            return 0, 0  # No reward or penalty if out-of-bounds

        demand = self.smart_grid.final_df['Total_current_demand'].iloc[self.smart_grid.time_step]
        total_previous_demand = self.smart_grid.final_df['Total_previous_demand'].iloc[self.smart_grid.time_step]
        solar_output = self.smart_grid.final_df['Solar kw/min'].iloc[self.smart_grid.time_step]
        supplied_energy = 0
        reward = 0
        penalty = 0

        # Scale the action to the demand
        action_scaled = action * demand

        # Battery usage
        battery_energy = 0
        if action_scaled[0] > 0 and action_scaled[1] > 0:  # Both charging and discharging
            penalty += 100  # High penalty for invalid actions
        else:
            if action_scaled[0] > 0:
                battery_energy = self.smart_grid.battery.battery_step('charging')
            elif action_scaled[1] > 0:
                battery_energy = self.smart_grid.battery.battery_step('discharging')
            supplied_energy += battery_energy
            reward -= battery_energy * self.smart_grid.battery.get_cost()  # Cost for battery usage

        # CHP usage
        chp_energy = action_scaled[2]
        reward -= self.smart_grid.chp.calculate_cost_at_current_step(chp_energy)
        supplied_energy += chp_energy

        # Electric market purchase and usage
        em_energy = action_scaled[3]
        if 6 <= self.smart_grid.time_step % 15 <= 9:  # Purchase between minutes 6 to 9
            self.smart_grid.em.make_purchase(em_energy)
        em_state = self.smart_grid.em.get_purchased_at_current_step()
        reward -= em_state['cost_of_electricity']
        supplied_energy += em_state['electricity_output']

        # Prior purchased and solar
        prior_purchased_energy = action_scaled[4]
        solar_energy = 0
        if action_scaled[5] > 0:
            solar_energy = solar_output

        reward -= total_previous_demand * prior_purchased_energy  # Cost for prior purchased
        reward -= solar_energy * self.solarcost  # Solar cost
        supplied_energy += prior_purchased_energy + solar_energy

        # Use public grid to meet any remaining demand
        remaining_demand = max(0, demand - supplied_energy)
        public_grid_energy = remaining_demand
        reward -= self.smart_grid.pg.get_price(public_grid_energy,
                                               demand) * 200  # Significant penalty for public grid usage
        supplied_energy += public_grid_energy

        # Ensure demand is met
        if supplied_energy < demand:
            penalty += (demand - supplied_energy) * 200  # Heavy penalty for not meeting demand

        # Penalize if the sum of utilization actions does not add up to 100%
        if not np.isclose(np.sum(action), 1):
            penalty += 50  # Arbitrary penalty for not adding up to 100%

        # Additional penalty for wrong actions (if applicable)
        if np.any(action < 0) or np.any(action > 1):
            penalty += 50  # Arbitrary penalty for invalid actions

        return reward, penalty





