import numpy as np
import pandas as pd


class RateManager:
    def __init__(self, previous_demand, base_rate, high_threshold=10, low_threshold=1, fluctuation=0.5):
        self.previous_demand = previous_demand
        self.base_rate = base_rate
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.fluctuation = fluctuation
        self.rate_df = None

    def generate_rates(self):
        print("Generating rates...")
        rates = []

        for demand in self.previous_demand:
            if demand < self.low_threshold:
                rate = self.base_rate * (1 - self.fluctuation * np.random.random())
            elif demand > self.high_threshold:
                rate = self.base_rate * (1 + self.fluctuation * np.random.random())
            else:
                rate = self.base_rate * (1 + (np.random.random() - 0.5) * self.fluctuation)
            rates.append(rate)

        self.rate_df = pd.DataFrame({'Rate': rates})

        #print("Generated Rates:")
        #print(self.rate_df.head(20))  # Print the first 20 rows for verification
        #self.save_to_excel('../../data/rate_schedule.xlsx')

        return self.rate_df

    def save_to_excel(self, file_path):
        if self.rate_df is not None:
            self.rate_df.to_excel(file_path, index=False)
        else:
            raise ValueError("Rate DataFrame is empty. Generate rates before saving to Excel.")

'''previous_day_demand_profile = [random.randint(0, 100) for _ in range(24 * 60)]

high_threshold = 70
low_threshold = 30

rate_manager = RateManager(previous_day_demand_profile,15,50, high_threshold=high_threshold, low_threshold=low_threshold)

current_rate = rate_manager.generate_rates()
rate_manager.save_to_excel()
    #print(f"Current Rate: {current_rate} rs/kWh")'''