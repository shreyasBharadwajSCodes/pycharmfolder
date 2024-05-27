import pandas as pd

from smart_grid_simulation.src.models.solar_model import SolarModel


class smartGridSystemDFPreprocessor:
    def __init__(self, simulatedfile, solarfile, solarcost_kwh):
        self.get_simulation_df(filename=simulatedfile)  # simulation_df_is_got
        self.simulation_total_demand()  # self.simulated_total_demand_df
        self.solar_data_get(solarfile=solarfile, solarcost_kwh=solarcost_kwh)
        self.combine_all_dfs()

    def solar_data_get(self, solarfile, solarcost_kwh):
        self.solarmodel = SolarModel(solar_file=solarfile,cost_per_kwh=solarcost_kwh)
        self.solardf = self.solarmodel.get_file(solarfile)

    def get_simulation_df(self, filename):
        self.simulated_load_df = pd.read_excel(filename)

    def simulation_total_demand(self):
        # Group by Time Step and sum the Current Demand and Previous Demand for each group
        grouped = self.simulated_load_df.groupby('Time Step').agg(
            {'Current Demand': 'sum', 'Previous Demand': 'sum'}).reset_index()

        # Create the DataFrame with the grouped data
        self.simulated_total_demand_df = pd.DataFrame(
            columns=['Time Step', 'Total_current_demand', 'Total_previous_demand'])
        self.simulated_total_demand_df['Time Step'] = grouped['Time Step']
        self.simulated_total_demand_df['Total_current_demand'] = grouped['Current Demand']
        self.simulated_total_demand_df['Total_previous_demand'] = grouped['Previous Demand']

        return self.simulated_total_demand_df

    def combine_all_dfs(self):
        self.solardf['Time'] = pd.to_datetime(self.solardf['Time'])

        # Calculate time steps by subtracting the minimum timestamp value and converting to minutes
        min_time = self.solardf['Time'].min()
        self.solardf['Time Step'] = (self.solardf['Time'] - min_time).dt.total_seconds() // 60 + 1

        # Drop the 'Time' column to reduce columns in the DataFrame
        self.solardf.drop('Time', axis=1, inplace=True)

        extra_row = pd.DataFrame({'Solar kw/min': [0.0], 'Time Step': [1441]})
        self.solardf = pd.concat([self.solardf, extra_row], ignore_index=True)

        self.final_df = pd.merge(self.simulated_total_demand_df, self.solardf, on='Time Step')

        return self.final_df

    def get_final_df(self):
        #returns final df and solar model , to be unpacked
        return (self.final_df,self.solarmodel)

'''
sgpre = smartGridSystemDFPreprocessor('../../data/load_details/simulation_20240521_185926_file.xlsx',
                                      '../../data/solar_generated_02-02-2024_10_0.15.xlsx', 5)
#print(sgpre.simulated_total_demand_df)
#print(sgpre.simulated_load_df)
#print(sgpre.solardf)
print(sgpre.final_df.to_string())'''
