import pandas as pd

from models.solar_model import SolarModel
from utils.load_profile import LoadProfileGenerator, LoadProfile

solar_parameters = {
    'latitude':21.1458,
    'longitude':79.0882,
    'efficiency':0.2,
    'num_cells':10,
    'date':'04-05-2024',
    'tz':'Asia/Kolkata'
}
solar_cost_per_kwh = 5

Paths = {
    'data' : '../data',
    'methods':'/methods',
    'models':'/models',
    'utils':'/utils',
    'tests':'../tests'
}

simulation_parameters = {'input_file': "../data/EKPC_hourly.csv",
                         'num_loads': 3,
                         }



if __name__ == '__main__':
    solar_model = SolarModel(cost_per_kwh=solar_cost_per_kwh)
    #solar_model.generate_solar_data_excel(solar_parameters)
    #solar_model.get_file("..\data\electricity_generated_04-05-2024_10_0.3.xlsx")
    #print(solar_model.calculate_cost_at_timestep(812))

