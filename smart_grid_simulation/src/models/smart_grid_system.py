from utils.smart_grid_system_preprocessor import smartGridSystemDFPreprocessor as preprocessor
from models.battery_model import BatteryModel
from models.public_grid_model import PublicGridModel
from models.chp_model import CHPModel
from models.prior_purchased_model import PriorPurchasedModel
from models.electric_market_model import ElectricMarket
from utils.rate_manager import RateManager

class SmartGridSystem:
    def __init__(self, sim_file, solar_file, solarcost_kwh):
        self.time_step = 0
        self.get_preprocessed_df(sim_file, solar_file, solarcost_kwh)
        self.battery = BatteryModel(capacity_kwh=10, max_charge_rate=5, max_discharge_rate=5,initial_soc=0.5)
        self.chp = CHPModel()
        self.rm = RateManager(previous_demand=self.final_df['Total_previous_demand'], base_rate=5)
        self.rate_df = self.rm.generate_rates()
        print(self.rate_df)
        self.em = ElectricMarket(self.rate_df)
        self.pg = PublicGridModel()
        self.ppm = PriorPurchasedModel(self.final_df['Total_previous_demand'])  # Given previous demand

    def get_preprocessed_df(self, sim_file, solar_file, solarcost_kwh):
        self.preprocessed = preprocessor(sim_file, solar_file, solarcost_kwh)
        self.final_df, self.solar_model = self.preprocessed.get_final_df()

    def get_state(self):
        return {
            'time step': self.time_step,
            'battery': self.battery.get_state(),
            'chp': self.chp.get_state(),
            'electric market': self.em.get_state(),
            'public grid': self.pg.get_state(),
            'prior purchased': self.ppm.get_state(self.time_step),
            'solar': self.final_df['Solar kw/min'].iloc[self.time_step],
            'DF': self.final_df.iloc[self.time_step].to_dict()  # Include current time step data
        }

    def reset(self):
        self.time_step = 0
        self.battery.reset()
        self.chp.reset()
        self.em.reset()
        self.pg.reset()
        self.ppm.reset()


#smart_grid_system = SmartGridSystem('../../data/load_details/simulation_file_20240523_150402.xlsx', '../../data/solar_generated_02-02-2024_10_0.15.xlsx', 0.1)
