import logging
import pandas as pd
import time
import matplotlib.pyplot as plt
from math import ceil
from multiprocessing import Pool, Manager
from smart_grid_simulation.src.models.smart_grid_system import SmartGridSystem

# Configure logging
logging.basicConfig(filename='Ans_optimization.log', level=logging.INFO)

# Load rate schedule
rate_df = pd.read_excel('rate_schedule.xlsx')

def is_possible_action(smart_grid_system, state, action, total_demand):
    """
    Determine if the action is feasible given the current state and demand.
    """
    battery_mode = action['battery_mode']
    battery_rate = action['battery_rate']
    chp_output = action['chp_output']
    solar_mode = action['solar_mode']
    electric_market_purchase = action['electric_market_purchase']
    public_grid_usage = action['public grid usage']

    total_produced = (
        (battery_rate if battery_mode == 'discharge' else -battery_rate if battery_mode == 'charge' else 0) +
        chp_output +
        (smart_grid_system.solar_model.get_output(state['time step']) if solar_mode == 'on' else 0) +
        electric_market_purchase
    )
    return total_produced + public_grid_usage >= total_demand

def beam_search_optimization(smart_grid_system, start_step, end_step, beam_width=5):
    current_state = smart_grid_system.get_state()
    current_state['total_cost'] = 0
    beam = [(current_state, [])]  # (state, actions)

    for t in range(start_step, end_step):
        logging.info(f"Time Step: {t}, Current Cost: {beam[0][0]['total_cost'] if beam else 'N/A'}")
        all_candidates = []

        for state, actions in beam:
            possible_acts = possible_actions(smart_grid_system, state, t)
            for action in possible_acts:
                if not is_possible_action(smart_grid_system, state, action, ceil(smart_grid_system.final_df['Total_current_demand'].iloc[t])):
                    continue
                next_state = transition(smart_grid_system, state, action)
                all_candidates.append((next_state, actions + [action]))

        # Sort all candidates by cost and keep the best ones
        all_candidates.sort(key=lambda x: x[0]['total_cost'])
        beam = all_candidates[:beam_width]

        if not beam:
            break

    return beam[0][0]['total_cost'], beam[0][1] if beam else (float('inf'), [])

def transition(smart_grid_system, state, action):
    next_state = state.copy()
    next_state['time step'] += 1
    smart_grid_system.battery.set_mode(action['battery_mode'])
    smart_grid_system.chp.set_output(action['chp_output'])
    smart_grid_system.solar_model.set_mode(action['solar_mode'])

    available_solar = smart_grid_system.solar_model.get_output(next_state['time step'])
    total_available_energy = available_solar + action['chp_output'] + action['electric_market_purchase'] + action['public grid usage']
    remaining_demand = max(0, smart_grid_system.final_df['Total_current_demand'].iloc[next_state['time step']] - total_available_energy)

    if action['battery_mode'] == 'charge':
        smart_grid_system.battery.charge(min(remaining_demand, total_available_energy))
    elif action['battery_mode'] == 'discharge':
        smart_grid_system.battery.discharge()

    next_state['battery'] = smart_grid_system.battery.get_state()
    next_state['chp'] = smart_grid_system.chp.get_state()
    next_state['public grid'] = smart_grid_system.pg.get_state()
    next_state['prior purchased'] = smart_grid_system.ppm.get_state(next_state['time step'])
    next_state['solar'] = available_solar
    next_state['total_cost'] = state.get('total_cost', 0) + action_cost(smart_grid_system, state, action)

    logging.info(f"Transitioned to next state: Time Step: {next_state['time step']}, Total Cost: {next_state['total_cost']}, Battery SOC: {next_state['battery'][0]}, CHP Output: {next_state['chp'][0]}, Solar Output: {available_solar}, Electric Market Purchase: {action['electric_market_purchase']}, Public Grid Usage: {action['public grid usage']}")

    return next_state

def action_cost(smart_grid_system, state, action):
    cost = 0
    cost += action['chp_output'] * smart_grid_system.chp.cost_per_kwh_electricity
    cost += smart_grid_system.solar_model.calculate_cost_at_timestep(state['time step'])

    if state['time step'] < len(rate_df):
        cost += rate_df.iloc[state['time step']]['Rate'] * action['electric_market_purchase']
    else:
        return float('inf')

    cost += smart_grid_system.pg.get_price(action['public grid usage'], smart_grid_system.final_df['Total_current_demand'].iloc[state['time step']])
    return cost

def possible_actions(smart_grid_system, state, time_step):
    actions = []
    battery_modes = ['idle', 'charge', 'discharge']
    chp_outputs = [0] + list(range(smart_grid_system.chp.min_operational_value, smart_grid_system.chp.max_operational_value + 1, 1))
    total_demand = ceil(smart_grid_system.final_df['Total_current_demand'].iloc[time_step])

    for battery_mode in battery_modes:
        battery_rate_range = range(1, smart_grid_system.battery.max_charge_rate + 1) if battery_mode == 'charge' else range(1, smart_grid_system.battery.max_discharge_rate + 1) if battery_mode == 'discharge' else [0]
        for battery_rate in battery_rate_range:
            for chp_output in chp_outputs:
                for electric_market_purchase in range(0, total_demand + 1, 1):  # Adjust the step size as needed
                    solar_mode = 'on' if smart_grid_system.solar_model.get_output(state['time step']) > 0 else 'off'
                    total_produced = (
                        (battery_rate if battery_mode == 'discharge' else -battery_rate if battery_mode == 'charge' else 0) +
                        chp_output +
                        (smart_grid_system.solar_model.get_output(state['time step']) if solar_mode == 'on' else 0) +
                        electric_market_purchase
                    )
                    public_grid_usage = max(0, total_demand - total_produced)

                    if total_produced + public_grid_usage >= total_demand:
                        actions.append({
                            'battery_mode': battery_mode,
                            'battery_rate': battery_rate,
                            'chp_output': chp_output,
                            'solar_mode': solar_mode,
                            'electric_market_purchase': electric_market_purchase,
                            'public grid usage': public_grid_usage
                        })

    return actions

def optimize_interval(args):
    smart_grid_system, start_step, end_step, beam_width, iter_value, result_dict = args
    try:
        logging.info(f"Optimizing interval from {start_step} to {end_step} with iter value {iter_value}...")
        result = beam_search_optimization(smart_grid_system, start_step, end_step, beam_width)
        logging.info(f"Interval from {start_step} to {end_step} optimized.")
        result_dict[iter_value] = result
    except Exception as e:
        logging.error(f"Error optimizing interval from {start_step} to {end_step}: {e}")
        result_dict[iter_value] = (float('inf'), [])
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
    minute_costs = []

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

        minute_costs.append(action_cost(smart_grid_system, state, action))

        total_costs.append(state['total_cost'])

    plt.figure(figsize=(20, 25))

    # Plot units consumed from each source
    plt.subplot(5, 1, 1)
    plt.plot(time_steps, battery_usage, label='Battery Usage')
    plt.plot(time_steps, chp_usage, label='CHP Usage')
    plt.plot(time_steps, solar_usage, label='Solar Usage')
    plt.plot(time_steps, electric_market_usage, label='Electric Market Usage')
    plt.plot(time_steps, public_grid_usage, label='Public Grid Usage')
    plt.ylabel('Units Consumed (kWh)')
    plt.title('Units Consumed from Each Source')
    plt.legend()

    # Plot total current demand
    plt.subplot(5, 1, 2)
    plt.plot(time_steps, total_current_demand, label='Total Current Demand')
    plt.ylabel('Units (kWh)')
    plt.title('Total Current Demand')
    plt.legend()

    # Plot total cost
    plt.subplot(5, 1, 3)
    plt.plot(time_steps, total_costs, label='Total Cost')
    plt.ylabel('Cost (R)')
    plt.title('Total Cost Over Time')
    plt.legend()

    # Plot costs from each source
    plt.subplot(5, 1, 4)
    plt.plot(time_steps, battery_costs, label='Battery Cost')
    plt.plot(time_steps, chp_costs, label='CHP Cost')
    plt.plot(time_steps, solar_costs, label='Solar Cost')
    plt.plot(time_steps, electric_market_costs, label='Electric Market Cost')
    plt.plot(time_steps, public_grid_costs, label='Public Grid Cost')
    plt.ylabel('Cost (p)')
    plt.title('Costs from Each Source')
    plt.legend()

    # Plot best result for each minute
    plt.subplot(5, 1, 5)
    plt.plot(time_steps, minute_costs, label='Cost per Minute')
    plt.ylabel('Cost (R)')
    plt.title('Cost of Electricity Bought Every Minute')
    plt.legend()

    plt.tight_layout()
    plt.savefig('smart_grid_optimization_results.png')

def distribute_procurement(actions):
    """Distribute the electric market procurement evenly across the billing interval."""
    billing_interval = 15  # minutes
    for i in range(0, len(actions), billing_interval):
        interval_actions = actions[i:i + billing_interval]
        if interval_actions:
            avg_purchase = sum(action['electric_market_purchase'] for action in interval_actions) / len(interval_actions)
            for action in interval_actions:
                action['electric_market_purchase'] = avg_purchase

# Example usage
if __name__ == "__main__":
    smart_grid_system = SmartGridSystem(sim_file=r'../../data/load_details/simulation_file_20240523_150250.xlsx',
                                        solar_file=r'../../data/solar_generated_02-05-2024_10_0.2.xlsx',
                                        solarcost_kwh=0.1)
    max_timesteps = 30  #1440 for full day
    interval_length = 30  # Shorter intervals for testing, e.g., 10 minutes

    # Split the optimization into intervals
    intervals = [(smart_grid_system, i, min(i + interval_length, max_timesteps), 5, i) for i in range(0, max_timesteps, interval_length)]

    print("Starting optimization...")
    logging.info("Starting optimization...")

    start_time = time.time()
    with Manager() as manager:
        result_dict = manager.dict()
        with Pool(processes=20) as pool:  # Adjust the number of processes based on your CPU cores
            pool.map(optimize_interval, [(smart_grid_system, start, end, beam_width, iter_value, result_dict) for smart_grid_system, start, end, beam_width, iter_value in intervals])

        # Combine results from all intervals
        results = [result_dict[i] for i in range(len(intervals)) if i in result_dict]
        total_cost = sum(result[0] for result in results)
        optimal_actions = [action for result in results for action in result[1]]

        # Distribute electric market procurement evenly across billing intervals
        distribute_procurement(optimal_actions)

        end_time = time.time()

        elapsed_time = end_time - start_time

        print("Optimization complete.")
        logging.info("Optimization complete.")
        print("Optimal Cost:", total_cost)
        print("Optimal Actions:", optimal_actions)
        print(f"Time taken for optimization: {elapsed_time:.2f} seconds")

        plot_graphs(optimal_actions, smart_grid_system)
