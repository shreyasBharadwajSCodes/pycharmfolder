import pandas as pd
import math
import heapq
import multiprocessing
import time
import matplotlib.pyplot as plt
from models.smart_grid_system import SmartGridSystem

# Load rate schedule
rate_df = pd.read_excel('rate_schedule.xlsx')

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
    next_state['total_cost'] = state.get('total_cost', 0) + action_cost(smart_grid_system, state, action)  # Corrected handling of total_cost

    return next_state

def action_cost(smart_grid_system, state, action):
    cost = 0
    cost += smart_grid_system.chp.calculate_cost_at_current_step()
    cost += smart_grid_system.solar_model.calculate_cost_at_timestep(state['time step'])

    # Ensure time step is within bounds for rate_df
    if state['time step'] < len(rate_df):
        cost += rate_df.iloc[state['time step']]['Rate'] * action['electric_market_purchase']
    else:
        # Skip the action if time step is out of bounds
        return float('inf')

    cost += smart_grid_system.pg.get_price(action['public grid usage'], state['DF']['Total_current_demand'])
    return cost

def look_ahead_cost(smart_grid_system, state, horizon=10):  # Reduced look-ahead window to 10 minutes
    total_estimated_cost = 0
    for i in range(horizon):
        possible_acts = possible_actions(smart_grid_system, state)
        if not possible_acts:
            break
        next_action = min(possible_acts, key=lambda a: action_cost(smart_grid_system, state, a))
        state = transition(smart_grid_system, state, next_action)
        total_estimated_cost += action_cost(smart_grid_system, state, next_action)
    return total_estimated_cost

def possible_actions(smart_grid_system, state):
    actions = []
    battery_modes = ['idle']
    if smart_grid_system.battery.soc > 0:
        battery_modes.append('discharge')
    if smart_grid_system.battery.soc < smart_grid_system.battery.capacity_kwh:
        battery_modes.append('charge')

    chp_outputs = [0] + list(range(smart_grid_system.chp.min_operational_value, smart_grid_system.chp.max_operational_value + 1))

    total_demand = state['DF']['Total_current_demand'] - state['prior purchased']
    public_grid_usage_without_solar = max(0, total_demand - smart_grid_system.battery.discharge(is_zero=1) - max(chp_outputs))

    for battery_mode in battery_modes:
        for chp_output in chp_outputs:
            solar_mode = 'on'  # Default to solar on
            if public_grid_usage_without_solar == 0:
                solar_mode = 'off'  # Only turn off if demand is fully satisfied without solar

            public_grid_usage = max(0, total_demand - smart_grid_system.battery.discharge(is_zero=1) - chp_output - (
                smart_grid_system.solar_model.get_output(state['time step']) if solar_mode == 'on' else 0))

            if public_grid_usage / total_demand <= 0.02:
                actions.append({
                    'battery_mode': battery_mode,
                    'chp_output': chp_output,
                    'solar_mode': solar_mode,
                    'electric_market_purchase': smart_grid_system.em.get_purchased_at_current_step()['electricity_output'],
                    'public grid usage': public_grid_usage  # Only add if condition is met
                })
            else:
                actions.append({
                    'battery_mode': battery_mode,
                    'chp_output': chp_output,
                    'solar_mode': solar_mode,
                    'electric_market_purchase': smart_grid_system.em.get_purchased_at_current_step()['electricity_output'],
                    'public grid usage': 0  # Set to 0 if condition is not met
                })
    return actions

def evaluate_action(args):
    smart_grid_system, current_state, action = args
    next_state = transition(smart_grid_system, current_state, action)
    next_cost = action_cost(smart_grid_system, current_state, action)
    estimated_future_cost = look_ahead_cost(smart_grid_system, next_state)
    total_estimated_cost = next_cost + estimated_future_cost
    return (total_estimated_cost, next_state, action)

def optimal_cost_multiprocessing(smart_grid_system, initial_state, num_workers=8, beam_width=10):
    priority_queue = []
    heapq.heappush(priority_queue, (0, id(initial_state), initial_state, []))  # (cost, id, state, actions_taken)
    min_cost = float('inf')
    best_actions = []
    visited = set()
    pool = multiprocessing.Pool(num_workers)

    iteration = 0
    while priority_queue:
        if iteration % beam_width == 0:
            # Only keep top beam_width states
            priority_queue = heapq.nlargest(beam_width, priority_queue)
            heapq.heapify(priority_queue)

        current_cost, _, current_state, actions_taken = heapq.heappop(priority_queue)

        if current_state['time step'] % 60 == 0:  # Log every hour
            print(f"Iteration {iteration}: Time Step {current_state['time step']} - Current Cost: {current_cost}")
        iteration += 1

        # If we have already visited this state with a lower cost, skip it
        state_key = (current_state['time step'], tuple(current_state['battery']), tuple(current_state['chp']),
                     tuple(current_state['electric market']), tuple(current_state['public grid']),
                     current_state['solar'])
        if state_key in visited:
            continue
        visited.add(state_key)

        if current_state['time step'] >= 24 * 60:  # End of the 24-hour period
            if current_cost < min_cost:
                min_cost = current_cost
                best_actions = actions_taken
            continue

        possible_acts = possible_actions(smart_grid_system, current_state)
        tasks = [(smart_grid_system, current_state, action) for action in possible_acts]
        results = pool.map(evaluate_action, tasks)

        for result in results:
            total_estimated_cost, next_state, action = result
            if total_estimated_cost < min_cost:
                heapq.heappush(priority_queue,
                               (total_estimated_cost, id(next_state), next_state, actions_taken + [action]))
                print(f"Queued next state with total estimated cost {total_estimated_cost}, action {action}")

    pool.close()
    pool.join()
    return min_cost, best_actions

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

    # Plot cost
    plt.subplot(4, 1, 3)
    plt.plot(time_steps, total_costs, label='Total Cost')
    plt.ylabel('Cost ($)')
    plt.title('Total Cost Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize the system
    smart_grid_system = SmartGridSystem(sim_file=r'../../data/load_details/simulation_file_20240523_150250.xlsx',
                                        solar_file=r'../../data/solar_generated_02-05-2024_10_0.2.xlsx',
                                        solarcost_kwh=0.1)
    initial_state = smart_grid_system.get_state()
    initial_state['total_cost'] = 0

    print("Starting optimization...")

    # Measure the time taken for optimization
    start_time = time.time()
    cost = 0
    optimal_actions = []
    try:
        cost, optimal_actions = optimal_cost_multiprocessing(smart_grid_system, initial_state)
    except IndexError as e:
        print(f"An error occurred: {e}")
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Optimization complete.")
    print("Optimal Cost:", cost)
    print("Optimal Actions:", optimal_actions)
    print(f"Time taken for optimization: {elapsed_time:.2f} seconds")

    # Plot the graphs
    plot_graphs(optimal_actions, smart_grid_system)
