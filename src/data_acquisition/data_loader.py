# src/data_acquisition/data_loader.py
"""
Data Ingestion & Feature Engineering Module.

Responsibilities:
1. Fetch raw meteorological data from Open-Meteo API.
2. Standardize column names to the internal schema.
3. Perform Temporal Feature Engineering (Cyclical Encoding) to transform
   linear timestamps into Neural Network-friendly formats (Sin/Cos).
"""

import requests
import os
import sys
import pandas as pd
import numpy as np
from src import config

def get_api_url() -> str:
    """
    Constructs the dynamic API endpoint based on the configuration settings.
    """
    return config.API_URL_TEMPLATE.format(
        lat=config.LOCATION['lat'],
        lon=config.LOCATION['lon'],
        start=config.API_START_DATE,
        end=config.API_END_DATE,
        timezone=config.LOCATION['timezone'].replace("/", "%2F")
    )

def download_data() -> None:
    """
    Downloads historical weather data with a caching mechanism.
    If the file already exists, it skips the download to save bandwidth.
    """
    if os.path.exists(config.RAW_DATA_PATH):
        print(f"Raw data cache found at: {config.RAW_DATA_PATH}")
        return

    url = get_api_url()
    print(f"Downloading data for {config.LOCATION['name']}...")
    print(f"Endpoint: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
        with open(config.RAW_DATA_PATH, 'wb') as f:
            f.write(response.content)

        print(f"Raw data saved.")

    except requests.exceptions.RequestException as e:
        print(f"API connection failed: {e}")
        sys.exit(1)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Temporal Feature Engineering.

    Converts the 'timestamp' index into cyclical continuous features using Sine/Cosine transformations.
    This allows the Neural Network to understand cyclical patterns (e.g., 23:00 is close to 00:00).

    New Features:
    - day_sin, day_cos: Captures Daily Cycle (Circadian rhythm).
    - year_sin, year_cos: Captures Annual Cycle (Seasonality).
    """
    df = df.copy()

    # Seconds in a day/year
    day = 24 * 60 * 60
    year = (365.2425) * day

    # Convert timestamp to seconds
    timestamp_s = df.index.map(pd.Timestamp.timestamp)

    # 1. Daily Cycle (Morning/Night)
    df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    # 2. Yearly Cycle (Summer/Winter)
    df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df

def load_raw_data() -> pd.DataFrame:
    """
    ETL Process: Extract, Transform, Load.

    1. Reads raw CSV.
    2. Maps column names to internal schema (including Precipitation).
    3. Sets Datetime Index.
    4. Computes Time Embeddings (Sin/Cos).
    5. Returns a DataFrame with exactly 9 Features + 1 Flag.
    """
    try:
        # Open-Meteo CSVs usually have 2 lines of metadata before the header
        df = pd.read_csv(config.RAW_DATA_PATH, skiprows=2)

        # Mapping raw API column names to our internal schema
        rename_map = {
            'time': 'timestamp',
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity',
            'surface_pressure (hPa)': 'pressure',
            'wind_speed_10m (m/s)': 'wind_speed',
            'precipitation (mm)': 'precipitation'
        }

        # Rename columns
        df.rename(columns=rename_map, inplace=True)

        # Datetime Conversion
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Validation: Check if all 5 physical parameters exist
        # We check the first 5 elements of FEATURE_COLS which correspond to physical data
        physical_cols = config.TARGET_COLS  # ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        for req in physical_cols:
            if req not in df.columns:
                raise ValueError(f"Missing required physical column '{req}' in raw data.")

        # Feature Engineering: Add the 4 time columns
        df = add_time_features(df)

        # Flag as real data
        df['is_simulated'] = 0

        return df[config.FEATURE_COLS + ['is_simulated']]

    except FileNotFoundError:
        print(f"Raw data file not found at {config.RAW_DATA_PATH}.")
        print("Run the download step first.")
        sys.exit(1)
    except Exception as e:
        print(f"Data loading pipeline failed: {e}")
        sys.exit(1)
