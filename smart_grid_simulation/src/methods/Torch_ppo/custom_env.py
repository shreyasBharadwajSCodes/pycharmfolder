import numpy as np
from gym import Env
from gym.spaces import Box

class CustomSmartGridEnv(Env):
    def __init__(self, smart_grid_system):
        self.smart_grid_system = smart_grid_system
        self.time_step = 0

        # Define action space: [battery action (discrete), chp action (continuous), market purchase action (continuous), solar action (discrete)]
        self.action_space = Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([2, 6, np.inf, 1]),
            dtype=np.float32
        )

        # Define observation space based on the state
        state = self._get_state()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(len(state),), dtype=np.float32)

    def reset(self):
        self.smart_grid_system.reset()
        self.time_step = 0
        return self._get_state()

    def step(self, action):
        action = np.array(action).flatten()  # Ensure action is a flat numpy array
        if action.shape[0] != 4:
            raise ValueError(f"Action array must have 4 elements, but got {action.shape[0]}")
        battery_action = int(action[0])
        chp_action = action[1]
        purchase_action = action[2]
        solar_action = int(action[3])

        self._apply_battery_action(battery_action)
        self._apply_chp_action(chp_action)
        self._apply_market_purchase_action(purchase_action)
        self._apply_solar_action(solar_action)

        costs = self._calculate_costs()
        penalties = self._calculate_penalties()

        reward = -costs - penalties

        self.time_step += 1
        done = self.time_step >= len(self.smart_grid_system.final_df)

        next_state = self._get_state()
        return next_state, reward, done, {}

    def _apply_battery_action(self, action):
        if action == 0:
            self.smart_grid_system.battery.set_mode("idle")
        elif action == 1:
            self.smart_grid_system.battery.set_mode("charge")
        elif action == 2:
            self.smart_grid_system.battery.set_mode("discharge")

    def _apply_chp_action(self, action):
        self.smart_grid_system.chp.set_output(action)

    def _apply_market_purchase_action(self, action):
        if 6 <= self.time_step % 15 <= 9:
            units = max(0, action)  # Ensure non-negative purchase
            self.smart_grid_system.em.make_purchase(units)

    def _apply_solar_action(self, action):
        if action == 0:
            self.smart_grid_system.solar_model.set_mode("off")
        elif action == 1:
            self.smart_grid_system.solar_model.set_mode("on")

    def _calculate_costs(self):
        timestep = self.time_step
        battery_cost = self.smart_grid_system.battery.get_cost()
        chp_cost = self.smart_grid_system.chp.calculate_cost_at_current_step()
        market_cost = self.smart_grid_system.em.get_price(timestep) * self.smart_grid_system.em.get_purchased_units()[timestep // 15]
        prior_purchased_cost = self.smart_grid_system.ppm.get_price(self.smart_grid_system.ppm.get_state(timestep))
        solar_cost = self.smart_grid_system.solar_model.calculate_cost_at_timestep(timestep)
        return battery_cost + chp_cost + market_cost + prior_purchased_cost + solar_cost

    def _calculate_penalties(self):
        penalty = 0
        total_demand = self.smart_grid_system.final_df['Total Demand'].iloc[self.time_step]
        demand_left = total_demand - sum([
            self.smart_grid_system.battery.get_state()[0],
            self.smart_grid_system.chp.current_output,
            self.smart_grid_system.solar_model.get_state(self.time_step)[0],
            self.smart_grid_system.ppm.get_state(self.time_step)
        ])
        penalty += self.smart_grid_system.pg.get_price(demand_left, total_demand)
        return penalty

    def _get_state(self):
        state = self.smart_grid_system.get_state()
        flat_state = []
        flat_state.append(state['time step'])
        flat_state.extend(state['battery'])
        flat_state.extend(state['chp'])
        flat_state.extend(state['electric market'])
        flat_state.extend(state['public grid'])
        flat_state.append(state['prior purchased'])  # Append single float value
        flat_state.append(state['solar'])
        flat_state.extend(state['DF'].values())
        # Ensure all elements are numeric
        flat_state = [float(x) for x in flat_state]
        return np.array(flat_state)
