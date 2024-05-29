import pandas as pd
import math
from collections import defaultdict

from models.smart_grid_system import SmartGridSystem

# Load rate schedule
rate_df = pd.read_excel('rate_schedule.xlsx')


# Define the transition and cost functions
def transition(smart_grid_system, state, action):
    next_state = state.copy()
    next_state['time step'] += 1  # Corrected key
    smart_grid_system.battery.set_mode(action['battery_mode'])
    smart_grid_system.chp.set_output(action['chp_output'])
    smart_grid_system.solar_model.set_mode(action['solar_mode'])

    available_solar = smart_grid_system.solar_model.get_output(next_state['time step'])
    battery_charge = smart_grid_system.battery.charge(available_solar)
    battery_discharge = smart_grid_system.battery.discharge(is_zero=1)  # Avoid double discharge

    next_state['battery'] = smart_grid_system.battery.get_state()
    next_state['chp'] = smart_grid_system.chp.get_state()
    next_state['electric market'] = smart_grid_system.em.get_state()
    next_state['public grid'] = smart_grid_system.pg.get_state()
    next_state['prior purchased'] = smart_grid_system.ppm.get_state(next_state['time step'])
    next_state['solar'] = available_solar
    next_state['total_cost'] = state.get('total_cost', 0) + action_cost(smart_grid_system, state,
                                                                        action)  # Corrected handling of total_cost

    return next_state


def action_cost(smart_grid_system, state, action):
    cost = 0
    cost += smart_grid_system.chp.calculate_cost_at_current_step()
    cost += smart_grid_system.solar_model.calculate_cost_at_timestep(state['time step'])
    cost += smart_grid_system.em.get_price(state['time step']) * action['electric_market_purchase']
    cost += smart_grid_system.pg.get_price(action['public grid usage'], state['DF']['Total_current_demand'])
    return cost


# Define possible actions based on the current state
def possible_actions(smart_grid_system, state):
    actions = []
    battery_modes = ['idle', 'charge', 'discharge']
    chp_outputs = [0] + list(
        range(smart_grid_system.chp.min_operational_value, smart_grid_system.chp.max_operational_value + 1))
    solar_modes = ['off', 'on']
    for battery_mode in battery_modes:
        for chp_output in chp_outputs:
            for solar_mode in solar_modes:
                actions.append({
                    'battery_mode': battery_mode,
                    'chp_output': chp_output,
                    'solar_mode': solar_mode,
                    'electric_market_purchase': smart_grid_system.em.get_purchased_at_current_step()[
                        'electricity_output'],
                    'public grid usage': max(0,
                                             state['DF']['Total_current_demand'] - smart_grid_system.battery.discharge(
                                                 is_zero=1) - chp_output - smart_grid_system.solar_model.get_output(
                                                 state['time step']))  # Corrected key
                })
    return actions


# Iterative function for optimal cost calculation
def optimal_cost(smart_grid_system, initial_state):
    state_queue = [(initial_state, [])]
    min_cost = float('inf')
    best_actions = []

    while state_queue:
        current_state, actions_taken = state_queue.pop(0)
        if current_state['time step'] >= 24 * 60:  # End of the 24-hour period
            current_cost = current_state['total_cost']
            if current_cost < min_cost:
                min_cost = current_cost
                best_actions = actions_taken
            continue

        for action in possible_actions(smart_grid_system, current_state):
            next_state = transition(smart_grid_system, current_state, action)
            state_queue.append((next_state, actions_taken + [action]))

    return min_cost, best_actions


# Example usage
if __name__ == "__main__":
    # Initialize the system
    smart_grid_system = SmartGridSystem(sim_file=r'../../data/load_details/simulation_file_20240523_150250.xlsx',
                                        solar_file=r'../../data/solar_generated_02-05-2024_10_0.2.xlsx',
                                        solarcost_kwh=0.1)
    smart_grid_system.rate_df = rate_df
    initial_state = smart_grid_system.get_state()
    initial_state['total_cost'] = 0
    cost, optimal_actions = optimal_cost(smart_grid_system, initial_state)

    print("Optimal Cost:", cost)
    print("Optimal Actions:", optimal_actions)
