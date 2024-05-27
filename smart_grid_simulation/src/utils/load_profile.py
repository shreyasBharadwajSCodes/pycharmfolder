import pandas as pd
import numpy as np
import random
from smart_grid_simulation.src.models.load_model import LoadModel

class LoadProfile:
    def __init__(self, breakdown_file, load_file):
        self.breakdown_file = breakdown_file
        self.load_file = load_file
        self.load_breakdown_data()
        self.load_load_data()
        self.adjust_demand()
        self.add_fluctuations()
        self.store_simulation_to_excel()

    def load_breakdown_data(self):
        # Load breakdown data from the breakdown file
        self.breakdown_df = pd.read_excel(self.breakdown_file)

    def load_load_data(self):
        # Load load data from the load file
        self.load_df = pd.read_excel(self.load_file)

    def adjust_demand(self):
        # Iterate through each breakdown row
        for index, row in self.breakdown_df.iterrows():
            load_name = row['Load Name']
            breakdown_minute = row['Breakdown Minute']
            recovery_time = row['Recovery Time']

            # Filter load rows for the current load and time range
            load_rows = self.load_df[(self.load_df['Load Name'] == load_name) &
                                      (self.load_df['Time Step'] >= breakdown_minute) &
                                      (self.load_df['Time Step'] < breakdown_minute + recovery_time)]

            # Set demand to 0 during the recovery period
            self.load_df.loc[load_rows.index, 'Current Demand'] = 0

    def add_fluctuations(self):
        # Add small fluctuations to the demand within random intervals of 5 to 15 minutes
        window_size = random.randint(5, 15)
        for index in range(0, len(self.load_df), window_size):
            # Apply fluctuations within the selected window
            current_window_size = min(window_size, len(self.load_df) - index)
            for i in range(index, index + current_window_size):
                if self.load_df.at[i, 'Current Demand'] != 0:
                    fluctuation = random.uniform(-0.1, 0.1) * self.load_df.at[i, 'Current Demand']
                    self.load_df.at[i, 'Current Demand'] += fluctuation
                else:
                    fluctuation = random.uniform(0,5)
                    self.load_df.at[i, 'Current Demand'] += abs(fluctuation)

    def store_simulation_to_excel(self):
        # Store the adjusted DataFrame in an Excel file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'../../data/load_details/simulation_file_{timestamp}.xlsx'
        self.load_df.to_excel(file_name, index=False)
        print(f"Simulation data saved to {file_name}")

class LoadProfileGenerator:
    def __init__(self, input_file, num_loads,bd_prob=0.005):
        self.input_file = input_file
        self.num_loads = num_loads
        self.bd_prob = bd_prob
        self.load_data()

    def load_data(self):
        # Read the input file containing dates and hourly demand values
        df = pd.read_csv(self.input_file)

        # Initialize load models list
        self.load_models = []

        # Randomly select two dates for each load and create current/previous demand profiles
        for load_num in range(1, self.num_loads + 1):
            load_dates = random.sample(df['Datetime'].str.split().str[0].unique().tolist(), 2)
            current_demand = df[df['Datetime'].str.split().str[0] == load_dates[0]]['Demand'].values
            previous_demand = df[df['Datetime'].str.split().str[0] == load_dates[1]]['Demand'].values

            # Extrapolate hourly demand to minute-by-minute data using np.repeat
            current_demand = np.repeat(current_demand, 60).tolist()
            previous_demand = np.repeat(previous_demand, 60).tolist()

            current_demand.append(current_demand[-1])
            previous_demand.append(previous_demand[-1])

            # Create LoadModel instance for each load
            load_name = f'Load_{load_num}'
            recovery_time = random.randint(30, 120)
            load_model = LoadModel(load_name, current_demand, previous_demand, recovery_time)

            # Add load model to the list
            self.load_models.append(load_model)
        # Generate breakdowns for each load
        self.generate_breakdowns()

        # Store the load profiles with breakdowns and recovery times in an Excel file
        self.store_breakdowns_to_excel()

        # Store load details with breakdowns
        self.store_load_details_with_breakdowns_to_excel()

    def generate_breakdowns(self):
        for load_model in self.load_models:
            recovery_complete = True  # Flag to track recovery completion
            for minute_num in range(len(load_model.current_demand)):
                if recovery_complete:
                    if random.random() <= self.bd_prob and minute_num > load_model.recovery_time:
                        breakdown_info = {'Minute': minute_num}
                        load_model.breakdowns.append(breakdown_info)
                        recovery_complete = False  # Set flag to False during breakdown
                else:
                    # Check if recovery is complete before allowing a new breakdown
                    if minute_num > load_model.recovery_time + load_model.breakdowns[-1]['Minute']:
                        recovery_complete = True

    def store_breakdowns_to_excel(self):
        # Initialize lists to store breakdown data including recovery time
        load_names = []
        breakdown_minutes = []
        recovery_times = []

        for load_model in self.load_models:
            for breakdown_info in load_model.breakdowns:
                load_names.append(load_model.load_name)
                breakdown_minutes.append(breakdown_info['Minute'])
                recovery_times.append(load_model.recovery_time)

        # Create DataFrame for breakdowns including recovery time
        breakdown_df = pd.DataFrame({
            'Load Name': load_names,
            'Breakdown Minute': breakdown_minutes,
            'Recovery Time': recovery_times
        })

        # Save breakdown DataFrame including recovery time to Excel file
        breakdown_file_name = '../../data/load_details/load_breakdowns.xlsx'
        breakdown_df.to_excel(breakdown_file_name, index=False)
        print(f"Breakdown information with recovery time saved to {breakdown_file_name}")

    def store_load_details_with_breakdowns_to_excel(self):
        # Initialize lists to store load details with breakdowns
        load_names = []
        time_steps = []
        current_demands = []
        previous_demands = []

        for load_model in self.load_models:
            # Repeat load name for each time step
            load_names.extend([load_model.load_name] * len(load_model.current_demand))

            # Append time steps
            time_steps.extend(range(1, len(load_model.current_demand) + 1))

            # Append current demand for the load
            current_demands.extend(load_model.current_demand)

            # Append previous demand for the load
            previous_demands.extend(load_model.previous_demand)

        # Create DataFrame from the lists
        load_details_df = pd.DataFrame({
            'Load Name': load_names,
            'Time Step': time_steps,
            'Current Demand': current_demands,
            'Previous Demand': previous_demands
        })
        load_details_df['Current Demand'] = load_details_df['Current Demand'].div(60)
        load_details_df['Previous Demand'] = load_details_df['Previous Demand'].div(60)
        # Save load details with breakdowns to Excel file
        load_details_file_name = '../../data/load_details/load_details_with_breakdowns.xlsx'
        load_details_df.to_excel(load_details_file_name, index=False)
        print(f"Load details with breakdowns saved to {load_details_file_name}")

    def get_load_data(self):
        # Return DataFrame and breakdown information dictionary
        return [self.df, self.breakdown_info]


input_file = "../../data/EKPC_hourly.csv"
num_loads = 5

generator = LoadProfileGenerator(input_file, num_loads)

breakdown_file = '../../data/load_details/load_breakdowns.xlsx'
load_file = '../../data/load_details/load_details_with_breakdowns.xlsx'

load_profile = LoadProfile(breakdown_file, load_file)
