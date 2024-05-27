import pandas as pd
from datetime import timedelta
from pvlib import irradiance, location
import requests
from requests.adapters import HTTPAdapter
import requests.packages.urllib3.util.retry as r


class SolarCellData:
    def __init__(self, latitude, longitude, efficiency, num_cells, date, tz='Asia/Kolkata'):
        self.latitude = latitude
        self.longitude = longitude
        self.efficiency = efficiency
        self.num_cells = num_cells
        self.date = date
        self.tz = tz

    def fetch_data_with_retry(self, url):
        retry_strategy = r.Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount('https://', adapter)
        session.mount('http://', adapter)

        return session.get(url, timeout=10)

    def get_cloud_cover_data(self):
        # Fetch cloud cover data from API with retry
        api_key = '3fa6065eb76f123c1cd5c418c5018d5d'
        start_date = self.date
        end_date = (pd.to_datetime(self.date) + timedelta(days=1)).strftime('%Y-%m-%d')
        url = f'http://api.openweathermap.org/data/2.5/forecast?id=524901&appid={api_key}&start={start_date}&end={end_date}&units=metric'
        response = self.fetch_data_with_retry(url)

        if response.status_code == 200:
            data = response.json()
            times = [pd.to_datetime(item['dt_txt']) for item in data['list']]
            cloud_cover = [item['clouds']['all'] / 100 for item in data['list']]

            # Create DataFrame with cloud cover data
            df = pd.DataFrame({
                'datetime': times,
                'cloud_cover': cloud_cover
            })
            df.set_index('datetime', inplace=True)

            # Resample and forward fill to match one-minute frequency
            df = df.resample('1min').ffill()
            df.reset_index(inplace=True)
            df['datetime'] = df['datetime'].dt.tz_localize(self.tz)  # Localize to specified timezone
            return df
        else:
            raise Exception(f"Failed to fetch cloud cover data. Status code: {response.status_code}")

    def get_irradiance_data(self):
        # Create location object
        site = location.Location(self.latitude, self.longitude, tz=self.tz)

        # Fetch cloud cover data
        cloud_cover_data = self.get_cloud_cover_data()

        # Create times with one-minute frequency
        times = pd.date_range(self.date, freq='1min', periods=60 * 24, tz=self.tz)

        # Calculate clear-sky GHI and transpose to plane of array
        clearsky = site.get_clearsky(times)
        solar_position = site.get_solarposition(times=times)

        # Adjust clear-sky irradiance based on real cloud cover data
        cloud_cover_interp = cloud_cover_data.set_index('datetime').reindex(times, method='nearest')
        clearsky['ghi'] *= (1 - cloud_cover_interp['cloud_cover'])
        clearsky['dni'] *= (1 - cloud_cover_interp['cloud_cover'])
        clearsky['dhi'] *= (1 - cloud_cover_interp['cloud_cover'])

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
            'Solar kw/min': electricity_generated
        })
        df['Solar kw/min'] = df['Solar kw/min'].div(60)

        # Convert timestamp to timezone-unaware
        df['Time'] = df['Time'].dt.tz_localize(None)

        return df

    def save_to_excel(self, df, excel_file_path):
        # Save the data to an Excel file
        df.to_excel(excel_file_path, index=False)
        print("Solar electricity generation data saved to Excel file:", excel_file_path)


'''
solar_data = SolarCellData(latitude=28.6139, longitude=77.209, efficiency=0.18, num_cells=100,
                           date='2024-05-03')
irradiance_df = solar_data.get_irradiance_data()
solar_data.save_to_excel(irradiance_df, 'solar_data.xlsx')
'''