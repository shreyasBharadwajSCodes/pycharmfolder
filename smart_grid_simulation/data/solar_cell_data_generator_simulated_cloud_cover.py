import pandas as pd
from datetime import timedelta
import pvlib
from pvlib import irradiance, location

class SolarCellData:
    def __init__(self, latitude, longitude, efficiency, num_cells, date, tz='Asia/Kolkata'):
        self.latitude = latitude
        self.longitude = longitude
        self.efficiency = efficiency
        self.num_cells = num_cells
        self.date = date
        self.tz = tz

    def get_cloud_cover_data(self):
        start_date = self.date
        end_date = (pd.to_datetime(self.date) + timedelta(days=1)).strftime('%Y-%m-%d')
        times = pd.date_range(start=start_date, end=end_date, freq='1min')
        cloud_cover = [0.2] * len(times)

        df = pd.DataFrame({
            'datetime': times,
            'cloud_cover': cloud_cover
        })
        df['datetime'] = df['datetime'].dt.tz_localize(self.tz)  # Localize to specified timezone
        return df

    def get_irradiance_data(self):
        # Create location object
        site = location.Location(self.latitude, self.longitude, tz=self.tz)

        # Fetch cloud cover data
        cloud_cover_data = self.get_cloud_cover_data()

        # Interpolate cloud cover data to match time intervals
        times = pd.date_range(self.date, freq='1min', periods=60 * 24, tz=self.tz)
        interpolated_cloud_cover = cloud_cover_data.set_index('datetime').reindex(times).interpolate().fillna(method='bfill')

        # Calculate clear-sky GHI and transpose to plane of array
        clearsky = site.get_clearsky(times)
        solar_position = site.get_solarposition(times=times)

        # Adjust clear-sky irradiance based on real cloud cover data
        clearsky['ghi'] *= (1 - interpolated_cloud_cover['cloud_cover'])
        clearsky['dni'] *= (1 - interpolated_cloud_cover['cloud_cover'])
        clearsky['dhi'] *= (1 - interpolated_cloud_cover['cloud_cover'])

        POA_irradiance = irradiance.get_total_irradiance(
            surface_tilt=25,  # Assuming a fixed tilt angle of 25 degrees
            surface_azimuth=180,  # Assuming a south-facing array
            dni=clearsky['dni'],
            ghi=clearsky['ghi'],
            dhi=clearsky['dhi'],
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'])

        # Calculate electricity generated per minute
        electricity_generated = POA_irradiance['poa_global'] * self.efficiency * self.num_cells

        # Create DataFrame with 'Time' and 'Electricity_generated (KW/min)' columns
        df = pd.DataFrame({
            'Time': times,
            'Electricity_generated (KW/min)': electricity_generated
        })

        # Convert timestamp to timezone-unaware
        df['Time'] = df['Time'].dt.tz_localize(None)

        return df

    def save_to_excel(self, df, excel_file_path):
        # Save the data to an Excel file
        df.to_excel(excel_file_path, index=False)
        print("Solar electricity generation data saved to Excel file:", excel_file_path)

#solar_data = SolarCellData(latitude=28.6139, longitude=77.209, efficiency=0.18, num_cells=100, date='2024-05-03')
#irradiance_df = solar_data.get_irradiance_data()
#solar_data.save_to_excel(irradiance_df, 'solar_data.xlsx')
