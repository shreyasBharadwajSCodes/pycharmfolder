import pandas as pd
import time
import matplotlib.pyplot as plt
from models.smart_grid_system import SmartGridSystem

# Load rate schedule
rate_df = pd.read_excel('rate_schedule.xlsx')

def greedy_optimization(smart_grid_system, max_timesteps=1440):
    current_state = smart_grid_system.get_state()
    current_state['total_cost'] = 0
    optimal_actions = []

    for t in range(max_timesteps):
        possible_acts = possible_actions(smart_grid_system, current_state)
        if not possible_acts:
            break
        next_action = min(possible_acts, key=lambda a: action_cost(smart_grid_system, current_state, a))
        current_state = transition(smart_grid_system, current_state, next_action)
        optimal_actions.append(next_action)
        if current_state['time step'] >= 24 * 60:
            break

    return current_state['total_cost'], optimal_actions

def transition(smart_grid_system, state, action):
    next_state = state.copy()
    next_state['time step'] += 1
    smart_grid_system.battery.set_mode(action['battery_mode'])
    smart_grid_system.chp.set_output(action['chp_output'])
    smart_grid_system.solar_model.set_mode(action['solar_mode'])

    available_solar = smart_grid_system.solar_model.get_output(next_state['time step'])
    smart_grid_system.battery.charge(available_solar)
    smart_grid_system.battery.discharge(is_zero=1)

    next_state['battery'] = smart_grid_system.battery.get_state()
    next_state['chp'] = smart_grid_system.chp.get_state()
    next_state['electric market'] = smart_grid_system.em.get_state()
    next_state['public grid'] = smart_grid_system.pg.get_state()
    next_state['prior purchased'] = smart_grid_system.ppm.get_state(next_state['time step'])
    next_state['solar'] = available_solar
    next_state['total_cost'] = state.get('total_cost', 0) + action_cost(smart_grid_system, state, action)

    return next_state

def action_cost(smart_grid_system, state, action):
    cost = 0
    cost += smart_grid_system.chp.calculate_cost_at_current_step()
    cost += smart_grid_system.solar_model.calculate_cost_at_timestep(state['time step'])

    if state['time step'] < len(rate_df):
        cost += rate_df.iloc[state['time step']]['Rate'] * action['electric_market_purchase']
    else:
        return float('inf')

    cost += smart_grid_system.pg.get_price(action['public grid usage'], state['DF']['Total_current_demand'])
    return cost

def possible_actions(smart_grid_system, state):
    actions = []
    battery_modes = ['idle']
    if smart_grid_system.battery.soc > 0:
        battery_modes.append('discharge')
    if smart_grid_system.battery.soc < smart_grid_system.battery.capacity_kwh:
        battery_modes.append('charge')

    chp_outputs = [0] + list(range(smart_grid_system.chp.min_operational_value, smart_grid_system.chp.max_operational_value + 1))
    total_demand = state['DF']['Total_current_demand'] - state['prior purchased']

    for battery_mode in battery_modes:
        for chp_output in chp_outputs:
            solar_mode = 'on'
            total_produced = (
                smart_grid_system.battery.discharge(is_zero=1) * (1 if battery_mode == 'discharge' else 0) +
                chp_output +
                (smart_grid_system.solar_model.get_output(state['time step']) if solar_mode == 'on' else 0)
            )
            public_grid_usage = max(0, total_demand - total_produced)

            actions.append({
                'battery_mode': battery_mode,
                'chp_output': chp_output,
                'solar_mode': solar_mode,
                'electric_market_purchase': smart_grid_system.em.get_purchased_at_current_step()['electricity_output'],
                'public grid usage': public_grid_usage if public_grid_usage / total_demand <= 0.02 else 0
            })

    return actions

def plot_graphs(actions, smart_grid_system):
    time_steps = range(len(actions))

    battery_usage = []
    chp_usage = []
    solar_usage = []
    electric_market_usage = []
    public_grid_usage = []
    total_costs = []
    total_current_demand = []

    state = smart_grid_system.get_state()
    state['total_cost'] = 0

    for action in actions:
        battery_usage.append(smart_grid_system.battery.soc)
        chp_usage.append(smart_grid_system.chp.current_output)
        solar_usage.append(smart_grid_system.solar_model.get_output(state['time step']))
        electric_market_usage.append(action['electric_market_purchase'])
        public_grid_usage.append(action['public grid usage'])
        total_costs.append(state['total_cost'])
        total_current_demand.append(state['DF']['Total_current_demand'])

        state = transition(smart_grid_system, state, action)

    plt.figure(figsize=(15, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time_steps, battery_usage, label='Battery Usage')
    plt.plot(time_steps, chp_usage, label='CHP Usage')
    plt.plot(time_steps, solar_usage, label='Solar Usage')
    plt.plot(time_steps, electric_market_usage, label='Electric Market Usage')
    plt.plot(time_steps, public_grid_usage, label='Public Grid Usage')
    plt.ylabel('Units Consumed (kWh)')
    plt.title('Units Consumed from Each Source')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time_steps, total_current_demand, label='Total Current Demand')
    plt.ylabel('Units (kWh)')
    plt.title('Total Current Demand')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time_steps, total_costs, label='Total Cost')
    plt.ylabel('Cost ($)')
    plt.title('Total Cost Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    smart_grid_system = SmartGridSystem(sim_file=r'../../data/load_details/simulation_file_20240523_150250.xlsx',
                                        solar_file=r'../../data/solar_generated_02-05-2024_10_0.2.xlsx',
                                        solarcost_kwh=0.1)
    max_timesteps = 500  # Adjust based on your time constraints

    print("Starting optimization...")

    start_time = time.time()
    cost, optimal_actions = greedy_optimization(smart_grid_system, max_timesteps=max_timesteps)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Optimization complete.")
    print("Optimal Cost:", cost)
    print("Optimal Actions:", optimal_actions)
    print(f"Time taken for optimization: {elapsed_time:.2f} seconds")

    plot_graphs(optimal_actions, smart_grid_system)
