import random
import math
import logging
import matplotlib.pyplot as plt

from models.smart_grid_system import SmartGridSystem

# Setup logging
logging.basicConfig(filename='SA_log.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def energy_function(allocation, smart_grid):
    total_cost = 0
    smart_grid.reset()
    for minute, usage in allocation.items():
        smart_grid.time_step = minute
        solar_value = smart_grid.final_df['Solar kw/min'].iloc[minute] if usage['solar'] > 0 else 0
        battery_value = smart_grid.battery.discharge(is_zero=1) if usage['battery']['mode'] == 'discharge' else 0
        utility_market_value = usage['utility_market']
        prior_purchased_value = smart_grid.ppm.get_state(minute)
        chp_value = usage['chp']
        public_grid_value = usage['public_grid']

        # Calculate total supply
        total_supply = (
            solar_value + battery_value + utility_market_value +
            prior_purchased_value + chp_value + public_grid_value
        )

        total_demand = smart_grid.final_df['Total_current_demand'].iloc[minute]

        if total_supply < total_demand:
            # Apply a heavy penalty if demand is not met
            total_cost += (total_demand - total_supply) * 1000

        # Calculate costs for each source
        solar_cost = solar_value * smart_grid.solar_model.calculate_cost_at_timestep(minute)
        battery_cost = smart_grid.battery.get_cost()
        utility_market_cost = utility_market_value * smart_grid.em.get_price(minute)
        prior_purchased_cost = smart_grid.ppm.get_price(minute)
        chp_cost = smart_grid.chp.calculate_cost_at_current_step()
        public_grid_cost = smart_grid.pg.get_price(public_grid_value, total_demand)

        total_cost += (
            solar_cost + battery_cost + utility_market_cost +
            prior_purchased_cost + chp_cost + public_grid_cost
        )

    return total_cost

def neighbor_function(current_allocation, smart_grid):
    new_allocation = current_allocation.copy()
    minute = random.choice(list(new_allocation.keys()))
    source = random.choice(list(new_allocation[minute].keys()))

    if source == "solar":
        # Set to 0 or the fixed value from final_df['Solar kw/min']
        new_allocation[minute][source] = 0 if new_allocation[minute][source] > 0 else \
        smart_grid.final_df['Solar kw/min'].iloc[minute]
    elif source == "battery":
        current_mode = new_allocation[minute][source]['mode']
        new_mode = random.choice(['idle', 'charge', 'discharge'])
        if new_mode == 'charge' and current_mode != 'charge':
            new_mode = 'charge'
        elif new_mode == 'discharge' and current_mode != 'discharge':
            new_mode = 'discharge'
        else:
            new_mode = 'idle'
        new_allocation[minute][source] = {'mode': new_mode}
    elif source == "utility_market":
        # Update the entire 15-minute interval
        interval_start = (minute // 15) * 15
        current_value = new_allocation[minute][source]
        purchase_value = random.uniform(max(0, current_value - 10), min(100, current_value + 10))
        for m in range(interval_start, interval_start + 15):
            if m in new_allocation:
                new_allocation[m][source] = purchase_value
    elif source == "prior_purchased":
        # Do nothing as prior purchased is fixed
        pass
    elif source == "chp":
        current_state = smart_grid.chp.get_state()
        if current_state[1]:  # if CHP is off
            if minute >= current_state[2]:
                new_allocation[minute][source] = random.uniform(10, 40)
        else:
            if random.random() > 0.5:
                new_allocation[minute][source] = 0
                # Set recovery time for CHP when turned off
                smart_grid.chp.turn_on_time = minute + 90
            else:
                current_output = new_allocation[minute][source]
                new_allocation[minute][source] = random.uniform(max(10, current_output - 5),
                                                                min(40, current_output + 5))
    elif source == "public_grid":
        current_value = new_allocation[minute][source]
        new_allocation[minute][source] = random.uniform(max(0, current_value - 0.1), min(2, current_value + 0.1))
    return new_allocation

def simulated_annealing(initial_solution, smart_grid, initial_temperature, cooling_rate, min_temperature, timesteps):
    current_solution = initial_solution
    current_energy = energy_function(current_solution, smart_grid)
    temperature = initial_temperature

    while temperature > min_temperature:
        new_solution = neighbor_function(current_solution, smart_grid)
        new_energy = energy_function(new_solution, smart_grid)
        delta_energy = new_energy - current_energy

        if delta_energy < 0 or random.uniform(0, 1) < math.exp(-delta_energy / temperature):
            current_solution = new_solution
            current_energy = new_energy

        temperature *= cooling_rate

        # Logging the current state and energy
        logging.info(f'Temperature: {temperature}, Current Energy: {current_energy}, Best Energy: {new_energy}')

        # Optional: Early termination based on a predefined number of timesteps
        if timesteps and timesteps > 0:
            timesteps -= 1
            if timesteps == 0:
                break

    return current_solution, current_energy

# Function to run the simulation for a specified number of timesteps
def run_simulation(timesteps=1440, sim_file='../../data/load_details/simulation_file_20240523_150402.xlsx',
                   solar_file='../../data/solar_generated_02-02-2024_10_0.15.xlsx'):
    smart_grid = SmartGridSystem(sim_file=sim_file, solar_file=solar_file, solarcost_kwh=0.05)

    # Initial solution setup for the specified number of timesteps
    initial_solution = {minute: {
        "solar": smart_grid.final_df['Solar kw/min'].iloc[minute],
        "battery": {'mode': 'idle'},
        "utility_market": 50,
        "prior_purchased": smart_grid.ppm.get_state(minute),  # Fixed value
        "chp": 30,
        "public_grid": 1} for minute in range(timesteps)
    }

    initial_temperature = 1000
    cooling_rate = 0.95
    min_temperature = 0.1

    best_solution, best_cost = simulated_annealing(
        initial_solution,
        smart_grid,
        initial_temperature,
        cooling_rate,
        min_temperature,
        timesteps
    )

    logging.info(f'Best Solution: {best_solution}')
    logging.info(f'Best Cost: {best_cost}')
    #print("Best Solution:", best_solution)
    #print("Best Cost:", best_cost)

    plot_results(best_solution, smart_grid)

def plot_results(best_solution, smart_grid):
    # Extract data from best_solution
    time_steps = list(best_solution.keys())
    solar_usage = [best_solution[t]['solar'] for t in time_steps]
    battery_usage = [
        smart_grid.battery.discharge(is_zero=1) if best_solution[t]['battery']['mode'] == 'discharge' else 0 for t in
        time_steps]
    utility_market_usage = [best_solution[t]['utility_market'] for t in time_steps]
    prior_purchased_usage = [best_solution[t]['prior_purchased'] for t in time_steps]
    chp_usage = [best_solution[t]['chp'] for t in time_steps]
    public_grid_usage = [best_solution[t]['public_grid'] for t in time_steps]

    total_demand = smart_grid.final_df['Total_current_demand'].iloc[:len(time_steps)]

    # Calculate total cost
    total_cost = [
        solar_cost + battery_cost + utility_market_cost + prior_purchased_cost + chp_cost + public_grid_cost
        for solar_cost, battery_cost, utility_market_cost, prior_purchased_cost, chp_cost, public_grid_cost
        in zip(
            solar_usage, battery_usage, utility_market_usage, prior_purchased_usage, chp_usage, public_grid_usage
        )
    ]

    # Plot Total Cost Over Time
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, total_cost, label='Total Cost')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Cost')
    plt.title('Total Cost Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Units of Electricity Used by Each Source Over Time
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, solar_usage, label='Solar')
    plt.plot(time_steps, battery_usage, label='Battery')
    plt.plot(time_steps, utility_market_usage, label='Utility Market')
    plt.plot(time_steps, prior_purchased_usage, label='Prior Purchased')
    plt.plot(time_steps, chp_usage, label='CHP')
    plt.plot(time_steps, public_grid_usage, label='Public Grid')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Units of Electricity (kWh)')
    plt.title('Electricity Usage by Source Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Total Demand Over Time
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, total_demand, label='Total Demand')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Total Demand (kWh)')
    plt.title('Total Demand Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Cost Breakdown by Source
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs[0, 0].plot(time_steps, solar_usage, label='Solar Cost')
    axs[0, 0].set_title('Solar Cost Over Time')
    axs[0, 1].plot(time_steps, battery_usage, label='Battery Cost')
    axs[0, 1].set_title('Battery Cost Over Time')
    axs[1, 0].plot(time_steps, utility_market_usage, label='Utility Market Cost')
    axs[1, 0].set_title('Utility Market Cost Over Time')
    axs[1, 1].plot(time_steps, prior_purchased_usage, label='Prior Purchased Cost')
    axs[1, 1].set_title('Prior Purchased Cost Over Time')
    axs[2, 0].plot(time_steps, chp_usage, label='CHP Cost')
    axs[2, 0].set_title('CHP Cost Over Time')
    axs[2, 1].plot(time_steps, public_grid_usage, label='Public Grid Cost')
    axs[2, 1].set_title('Public Grid Cost Over Time')

    for ax in axs.flat:
        ax.set(xlabel='Time (minutes)', ylabel='Cost')
        ax.label_outer()
        ax.legend()
        ax.grid(True)

    plt.show()

# Example usage:
# Run the simulation for 1440 timesteps
run_simulation(timesteps=1440)
