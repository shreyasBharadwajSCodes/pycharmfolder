import pandas as pd
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool
from models.smart_grid_system import SmartGridSystem

# Load rate schedule
rate_df = pd.read_excel('rate_schedule.xlsx')

def hybrid_greedy_optimization(smart_grid_system, start_step, end_step, look_ahead_window=3):
    current_state = smart_grid_system.get_state()
    current_state['total_cost'] = 0
    optimal_actions = []

    for t in range(start_step, end_step):
        print(f"Time Step: {t}, Current Cost: {current_state['total_cost']}")
        possible_acts = possible_actions(smart_grid_system, current_state, t)
        if not possible_acts:
            break
        next_action = min(possible_acts, key=lambda a: action_cost(smart_grid_system, current_state, a) + look_ahead_cost(smart_grid_system, current_state, a, look_ahead_window))
        current_state = transition(smart_grid_system, current_state, next_action)
        optimal_actions.append(next_action)

        if current_state['time step'] >= end_step:
            break

    return current_state['total_cost'], optimal_actions

def look_ahead_cost(smart_grid_system, state, action, horizon):
    future_cost = 0
    next_state = transition(smart_grid_system, state, action)
    for i in range(horizon):
        print(f"Look-Ahead Step: {i}, Current Cost: {state['total_cost']}")
        possible_acts = possible_actions(smart_grid_system, next_state, next_state['time step'])
        if not possible_acts:
            break
        next_action = min(possible_acts, key=lambda a: action_cost(smart_grid_system, next_state, a))
        future_cost += action_cost(smart_grid_system, next_state, next_action)
        next_state = transition(smart_grid_system, next_state, next_action)
    return future_cost

def transition(smart_grid_system, state, action):
    next_state = state.copy()
    next_state['time step'] += 1
    smart_grid_system.battery.set_mode(action['battery_mode'])
    smart_grid_system.chp.set_output(action['chp_output'])
    smart_grid_system.solar_model.set_mode(action['solar_mode'])

    available_solar = smart_grid_system.solar_model.get_output(next_state['time step'])
    total_available_energy = available_solar + action['chp_output'] + action['electric_market_purchase'] + action['public grid usage']
    battery_charge = smart_grid_system.battery.charge(total_available_energy)
    battery_discharge = smart_grid_system.battery.discharge(is_zero=1)

    next_state['battery'] = smart_grid_system.battery.get_state()
    next_state['chp'] = smart_grid_system.chp.get_state()
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

    cost += smart_grid_system.pg.get_price(action['public grid usage'], smart_grid_system.final_df['Total_current_demand'].iloc[state['time step']])
    return cost

def possible_actions(smart_grid_system, state, time_step):
    actions = []
    battery_modes = ['idle']
    if smart_grid_system.battery.soc > 0:
        battery_modes.append('discharge')
    if smart_grid_system.battery.soc < smart_grid_system.battery.capacity_kwh:
        battery_modes.append('charge')

    chp_outputs = [0] + list(range(smart_grid_system.chp.min_operational_value, smart_grid_system.chp.max_operational_value + 1, 1))
    total_demand = smart_grid_system.final_df['Total_current_demand'].iloc[time_step] - state['prior purchased']

    for battery_mode in battery_modes:
        for chp_output in chp_outputs:
            for electric_market_purchase in range(0, int(total_demand), 1):  # Adjust the step size as needed
                solar_mode = 'on'
                total_produced = (
                    smart_grid_system.battery.discharge(is_zero=1) * (1 if battery_mode == 'discharge' else 0) +
                    chp_output +
                    (smart_grid_system.solar_model.get_output(state['time step']) if solar_mode == 'on' else 0) +
                    electric_market_purchase
                )
                public_grid_usage = max(0, total_demand - total_produced)

                # Ensure demand satisfaction
                if total_produced + public_grid_usage >= total_demand:
                    if public_grid_usage / total_demand <= 0.02:
                        actions.append({
                            'battery_mode': battery_mode,
                            'chp_output': chp_output,
                            'solar_mode': solar_mode,
                            'electric_market_purchase': electric_market_purchase,
                            'public grid usage': public_grid_usage
                        })
                    else:
                        actions.append({
                            'battery_mode': battery_mode,
                            'chp_output': chp_output,
                            'solar_mode': solar_mode,
                            'electric_market_purchase': electric_market_purchase,
                            'public grid usage': 0
                        })
                elif public_grid_usage > 0:  # Use public grid if required to meet demand
                    actions.append({
                        'battery_mode': battery_mode,
                        'chp_output': chp_output,
                        'solar_mode': solar_mode,
                        'electric_market_purchase': electric_market_purchase,
                        'public grid usage': public_grid_usage
                    })

    return actions

def optimize_interval(args):
    smart_grid_system, start_step, end_step, look_ahead_window = args
    print(f"Optimizing interval from {start_step} to {end_step}...")
    result = hybrid_greedy_optimization(smart_grid_system, start_step, end_step, look_ahead_window)
    print(f"Interval from {start_step} to {end_step} optimized.")
    return result

def plot_graphs(actions, smart_grid_system):
    time_steps = range(len(actions))

    battery_usage = []
    chp_usage = []
    solar_usage = []
    electric_market_usage = []
    public_grid_usage = []
    total_costs = []
    total_current_demand = []
    battery_costs = []
    chp_costs = []
    solar_costs = []
    electric_market_costs = []
    public_grid_costs = []

    state = smart_grid_system.get_state()
    state['total_cost'] = 0

    for action in actions:
        battery_usage.append(smart_grid_system.battery.soc)
        chp_usage.append(smart_grid_system.chp.current_output)
        solar_usage.append(smart_grid_system.solar_model.get_output(state['time step']))
        electric_market_usage.append(action['electric_market_purchase'])
        public_grid_usage.append(action['public grid usage'])
        total_current_demand.append(smart_grid_system.final_df['Total_current_demand'].iloc[state['time step']])

        state = transition(smart_grid_system, state, action)

        battery_costs.append(smart_grid_system.battery.get_cost())
        chp_costs.append(smart_grid_system.chp.calculate_cost_at_current_step())
        solar_costs.append(smart_grid_system.solar_model.calculate_cost_at_timestep(state['time step']))
        electric_market_costs.append(rate_df.iloc[state['time step']]['Rate'] * action['electric_market_purchase'] if state['time step'] < len(rate_df) else 0)
        public_grid_costs.append(smart_grid_system.pg.get_price(action['public grid usage'], smart_grid_system.final_df['Total_current_demand'].iloc[state['time step']]))

        total_costs.append(state['total_cost'])

    plt.figure(figsize=(15, 15))

    # Plot units consumed from each source
    plt.subplot(4, 1, 1)
    plt.plot(time_steps, battery_usage, label='Battery Usage')
    plt.plot(time_steps, chp_usage, label='CHP Usage')
    plt.plot(time_steps, solar_usage, label='Solar Usage')
    plt.plot(time_steps, electric_market_usage, label='Electric Market Usage')
    plt.plot(time_steps, public_grid_usage, label='Public Grid Usage')
    plt.ylabel('Units Consumed (kWh)')
    plt.title('Units Consumed from Each Source')
    plt.legend()

    # Plot total current demand
    plt.subplot(4, 1, 2)
    plt.plot(time_steps, total_current_demand, label='Total Current Demand')
    plt.ylabel('Units (kWh)')
    plt.title('Total Current Demand')
    plt.legend()

    # Plot total cost
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, total_costs, label='Total Cost')
    plt.ylabel('Cost (R)')
    plt.title('Total Cost Over Time')
    plt.legend()

    # Plot costs from each source
    plt.subplot(4, 1, 4)
    plt.plot(time_steps, battery_costs, label='Battery Cost')
    plt.plot(time_steps, chp_costs, label='CHP Cost')
    plt.plot(time_steps, solar_costs, label='Solar Cost')
    plt.plot(time_steps, electric_market_costs, label='Electric Market Cost')
    plt.plot(time_steps, public_grid_costs, label='Public Grid Cost')
    plt.ylabel('Cost (R)')
    plt.title('Costs from Each Source')
    plt.legend()

    plt.tight_layout()
    plt.show()

def distribute_procurement(actions):
    """Distribute the electric market procurement evenly across the billing interval."""
    billing_interval = 15  # minutes
    for i in range(0, len(actions), billing_interval):
        interval_actions = actions[i:i+billing_interval]
        if interval_actions:
            avg_purchase = sum(action['electric_market_purchase'] for action in interval_actions) / len(interval_actions)
            for action in interval_actions:
                action['electric_market_purchase'] = avg_purchase

# Example usage
if __name__ == "__main__":
    smart_grid_system = SmartGridSystem(sim_file=r'../../data/load_details/simulation_file_20240523_150250.xlsx',
                                        solar_file=r'../../data/solar_generated_02-05-2024_10_0.2.xlsx',
                                        solarcost_kwh=0.1)
    max_timesteps = 30  # Change this to 30 for testing, 1440 for full day
    interval_length = 10  # Shorter intervals for testing, e.g., 10 minutes
    look_ahead_window = 3

    # Split the optimization into intervals
    intervals = [(smart_grid_system, i, min(i + interval_length, max_timesteps), look_ahead_window) for i in range(0, max_timesteps, interval_length)]

    print("Starting optimization...")

    start_time = time.time()
    with Pool(processes=4) as pool:  # Adjust the number of processes based on your CPU cores
        results = pool.map(optimize_interval, intervals)

    # Combine results from all intervals
    total_cost = sum(result[0] for result in results)
    optimal_actions = [action for result in results for action in result[1]]

    # Distribute electric market procurement evenly across billing intervals
    distribute_procurement(optimal_actions)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print("Optimization complete.")
    print("Optimal Cost:", total_cost)
    print("Optimal Actions:", optimal_actions)
    print(f"Time taken for optimization: {elapsed_time:.2f} seconds")

    plot_graphs(optimal_actions, smart_grid_system)
