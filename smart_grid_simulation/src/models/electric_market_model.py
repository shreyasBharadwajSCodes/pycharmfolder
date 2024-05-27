import pandas as pd
'''
class ElectricMarket:
    def __init__(self, rate_df):
        self.rate_df = rate_df
        self.current_minute = 0
        self.purchased_units = [0] * ((len(rate_df) // 15) + 1)  # Initialize purchased units for every 15 minutes

    def make_purchase(self, units):
        interval_index = (self.current_minute + 1) // 15  # Purchase takes effect in the next interval
        if interval_index < len(self.purchased_units):
            if 6 <= self.current_minute % 15 <= 9:  # Purchase between minutes 6 to 9
                self.purchased_units[interval_index] += units  # Reflect purchase in current 15-minute interval

    def get_purchased_units(self):
        return self.purchased_units

    def get_price(self, timestep):
        if 0 <= timestep < len(self.rate_df):
            rate = self.rate_df.iloc[timestep]['Rate']
            return rate
        return 0

    def get_state(self):
        if self.current_minute < len(self.rate_df):
            rate = self.rate_df.iloc[self.current_minute]['Rate']
        else:
            rate = 0
        self.current_minute += 1
        return [rate]

    def get_purchased_at_current_step(self):
        interval_index = self.current_minute // 15
        if 0 <= interval_index < len(self.purchased_units):
            purchased_units = self.purchased_units[interval_index]
            rate = self.get_price(self.current_minute)
            cost = purchased_units * rate
            return {'electricity_output': purchased_units, 'cost_of_electricity': cost}
        return {'electricity_output': 0, 'cost_of_electricity': 0}

    def reset(self):
        self.current_minute = 0
        self.purchased_units = [0] * ((len(self.rate_df) // 15) + 1)
class ElectricMarket:
    def __init__(self, rate_df):
        self.rate_df = rate_df
        self.current_minute = 0
        self.purchased_units = [0] * ((len(rate_df) // 15) + 1)  # Initialize purchased units for every 15 minutes

    def make_purchase(self, units):
        interval_index = (self.current_minute + 1) // 15  # Purchase takes effect in the next interval
        if interval_index < len(self.purchased_units):
            if 6 <= self.current_minute % 15 <= 9:  # Purchase between minutes 6 to 9
                self.purchased_units[interval_index] += units  # Reflect purchase in current 15-minute interval

    def get_purchased_units(self):
        return self.purchased_units

    def get_price(self, timestep):
        if 0 <= timestep < len(self.rate_df):
            rate = self.rate_df.iloc[timestep]['Rate']
            return rate
        return 0

    def get_state(self):
        if self.current_minute < len(self.rate_df):
            rate = self.rate_df.iloc[self.current_minute]['Rate']
        else:
            rate = 0
        self.current_minute += 1
        return [rate]

    def get_purchased_at_current_step(self):
        interval_index = self.current_minute // 15
        if 0 <= interval_index < len(self.purchased_units):
            purchased_units = self.purchased_units[interval_index]
            rate = self.get_price(self.current_minute)
            cost = purchased_units * rate
            return {'electricity_output': purchased_units, 'cost_of_electricity': cost}
        return {'electricity_output': 0, 'cost_of_electricity': 0}

    def reset(self):
        self.current_minute = 0
        self.purchased_units = [0] * ((len(self.rate_df) // 15) + 1)'''
import pandas as pd

import pandas as pd

class ElectricMarket:
    def __init__(self, rate_df):
        self.rate_df = rate_df
        self.current_minute = 0
        self.purchased_units = [0] * ((len(rate_df) // 15) + 1)

    def make_purchase(self, units):
        interval_index = (self.current_minute + 1) // 15
        if interval_index < len(self.purchased_units):
            if 6 <= self.current_minute % 15 <= 9:
                self.purchased_units[interval_index] += units

    def get_purchased_units(self):
        return self.purchased_units

    def get_price(self, timestep):
        if 0 <= timestep < len(self.rate_df):
            rate = self.rate_df.iloc[timestep]['Rate']
            return rate
        return 0

    def get_state(self):
        if self.current_minute < len(self.rate_df):
            rate = self.rate_df.iloc[self.current_minute]['Rate']
        else:
            rate = 0
        self.current_minute += 1
        return [rate]

    def get_purchased_at_current_step(self):
        interval_index = self.current_minute // 15
        if 0 <= interval_index < len(self.purchased_units):
            purchased_units = self.purchased_units[interval_index]
            rate = self.get_price(self.current_minute)
            cost = purchased_units * rate
            return {'electricity_output': purchased_units, 'cost_of_electricity': cost}
        return {'electricity_output': 0, 'cost_of_electricity': 0}

    def reset(self):
        self.current_minute = 0
        self.purchased_units = [0] * ((len(self.rate_df) // 15) + 1)
