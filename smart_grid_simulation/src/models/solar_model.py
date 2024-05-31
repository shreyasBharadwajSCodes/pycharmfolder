'''import math
import pandas as pd
from smart_grid_simulation.data.solar_cell_data_generator import SolarCellData

class SolarModel:
    def __init__(self, cost_per_kwh):
        self.solar_df = None
        self.cost_per_kwh = cost_per_kwh
        self.excel_file_path = ''
        self.mode = 'off'  # Mode can be 'on' or 'off'

    def set_mode(self, mode):
        if mode not in ['on', 'off']:
            raise ValueError("Invalid mode for solar.")
        self.mode = mode

    def calculate_cost_at_timestep(self, i):
        if self.solar_df is None:
            raise ValueError("Solar DF not loaded.")
        i = min(i, len(self.solar_df) - 1)  # Ensure index is within bounds

        if self.mode == 'off':
            return 0

        electricity_generated = self.solar_df.iloc[i]['Solar kw/min']
        cost_per_minute = electricity_generated * self.cost_per_kwh / 60
        return math.ceil(cost_per_minute)

    def generate_solar_data_excel(self, solar_parameters: dict, path='../data/solar_generated'):
        solar_data = SolarCellData(
            solar_parameters['latitude'],
            solar_parameters['longitude'],
            solar_parameters['efficiency'],
            solar_parameters['num_cells'],
            solar_parameters['date'],
            solar_parameters['tz']
        )

        df = solar_data.get_irradiance_data()

        self.excel_file_path = f"{path}_{solar_parameters['date']}_{solar_parameters['num_cells']}_{solar_parameters['efficiency']}.xlsx"
        solar_data.save_to_excel(df, self.excel_file_path)

        self.solar_df = df  # Store it as solar_df

        return df  # Return the df containing solar data

    def get_file(self, path):
        try:
            self.solar_df = pd.read_excel(path)
            self.excel_file_path = path
            return self.solar_df
        except Exception as e:
            print(e)

    def get_output(self, time_step):
        if self.solar_df is None or time_step >= len(self.solar_df):
            return 0
        return self.solar_df.iloc[time_step]['Solar kw/min']

    def get_state(self, time_step):
        return [self.get_output(time_step) if self.mode == 'on' else 0]

    def reset(self):
        self.mode = 'off'
'''
import math

import pandas as pd

from smart_grid_simulation.data.solar_cell_data_generator import SolarCellData


class SolarModel:
    def __init__(self, solar_file, cost_per_kwh):
        self.solar_file = solar_file
        self.solar_df = pd.read_excel(solar_file)
        self.cost_per_kwh = cost_per_kwh
        self.mode = 'off'

    def set_mode(self, mode):
        if mode not in ['off', 'on']:
            raise ValueError("Invalid mode for solar.")
        self.mode = mode

    def calculate_cost_at_timestep(self, i):
        if self.solar_df is None:
            raise ValueError("Solar DF not loaded.")
        i = min(i, len(self.solar_df) - 1)  # Ensure index is within bounds

        if self.mode == 'off':
            return 0

        electricity_generated = self.solar_df.iloc[i]['Solar kw/min']
        cost_per_minute = electricity_generated * self.cost_per_kwh / 60
        return math.ceil(cost_per_minute)

    def generate_solar_data_excel(self, solar_parameters: dict, path='../data/solar_generated'):
        solar_data = SolarCellData(
            solar_parameters['latitude'],
            solar_parameters['longitude'],
            solar_parameters['efficiency'],
            solar_parameters['num_cells'],
            solar_parameters['date'],
            solar_parameters['tz']
        )

        df = solar_data.get_irradiance_data()

        self.excel_file_path = f"{path}_{solar_parameters['date']}_{solar_parameters['num_cells']}_{solar_parameters['efficiency']}.xlsx"
        solar_data.save_to_excel(df, self.excel_file_path)

        self.solar_df = df  # Store it as solar_df

        return df  # Return the df containing solar data

    def get_file(self, path):
        try:
            self.solar_df = pd.read_excel(path)
            self.excel_file_path = path
            return self.solar_df
        except Exception as e:
            print(e)


    def get_output(self, timestep):
        if self.mode == 'off':
            return 0
        if timestep < len(self.solar_df):
            return self.solar_df.iloc[timestep]['Solar kw/min']
        return 0

    def get_state(self, timestep):
        return [self.get_output(timestep)]

    def reset(self):
        self.mode = 'off'

'''
    def calculate_cost_at_timestep(self, timestep):
        if self.mode == 'off':
            return 0
        return self.get_output(timestep) * self.cost_per_kwh
'''