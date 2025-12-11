# src/data_acquisition/data_loader.py
import requests
import os
import sys
import pandas as pd
from src import config  # Import configuration


def get_api_url() -> str:
    """Builds the dynamic API URL based on config location."""
    return config.API_URL_TEMPLATE.format(
        lat=config.LOCATION['lat'],
        lon=config.LOCATION['lon'],
        start=config.API_START_DATE,
        end=config.API_END_DATE,
        timezone=config.LOCATION['timezone'].replace("/", "%2F")
    )


def download_data() -> None:
    """Downloads raw weather data from Open-Meteo API."""
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"[INFO] Raw data found at: {config.RAW_DATA_PATH}")
        return

    url = get_api_url()
    print(f"[INFO] Downloading data for {config.LOCATION['name']}...")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    with open(config.RAW_DATA_PATH, 'wb') as f:
        f.write(response.content)
    print(f"[SUCCESS] Raw data saved.")


def load_raw_data() -> pd.DataFrame:
    """Utility to load raw CSV into DataFrame."""
    try:
        df = pd.read_csv(config.RAW_DATA_PATH, skiprows=2)
        # Rename columns standard
        rename_map = {
            'time': 'timestamp',
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity',
            'surface_pressure (hPa)': 'pressure',
            'wind_speed_10m (m/s)': 'wind_speed'
        }
        df.rename(columns=rename_map, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        # Add flag for real data
        df['is_simulated'] = 0
        return df
    except Exception as e:
        print(f"[ERROR] Could not load raw data: {e}")
        sys.exit(1)