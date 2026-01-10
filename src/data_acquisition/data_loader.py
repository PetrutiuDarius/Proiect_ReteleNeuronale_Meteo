# src/data_acquisition/data_loader.py
"""
Data ingestion module.

Handles the extraction of raw meteorological data from the Open-Meteo API.
It standardizes column names to ensure downstream compatibility with the
5-parameter Neural Network architecture.
"""

import requests
import os
import sys
import pandas as pd
from typing import NoReturn
from src import config

def get_api_url() -> str:
    """
    Constructs the dynamic API endpoint based on configuration settings..
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
    Fetches historical weather data and saves it locally.
    Implements a caching mechanism: skips download if the raw file exists.
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
    except requests.exceptions.RequestException as e:
        print(f"API Download failed: {e}")
        sys.exit(1)

    # Ensure directory hierarchy exists
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)

    with open(config.RAW_DATA_PATH, 'wb') as f:
        f.write(response.content)

    print(f"Raw dataset acquired and saved.")

def load_raw_data() -> pd.DataFrame:
    """
    Loads and cleans the raw CSV data.

    Transformation Steps:
    1. Skips metadata rows.
    2. Renames columns to standard English identifiers.
    3. Enforces datetime indexing.
    4. Initializes the 'is_simulated' flag (0 for real data).

    Returns:
        pd.DataFrame: The standardized dataframe ready for augmentation.
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

        # Verify all required columns exist before renaming
        # This prevents obscure KeyErrors later if the API format changes
        for col in rename_map.keys():
            if col not in df.columns:
                # Fallback check for slight naming variations if necessary
                pass

        df.rename(columns=rename_map, inplace=True)

        # Validate schema
        required_cols = config.FEATURE_COLS  # ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        for req in required_cols:
            if req not in df.columns:
                raise ValueError(f"Missing required column '{req}' in raw data.")

        # Temporal indexing
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Flag as real data
        df['is_simulated'] = 0

        return df

    except FileNotFoundError:
        print(f"Raw data file not found at {config.RAW_DATA_PATH}.")
        print("Run the download step first.")
        sys.exit(1)
    except Exception as e:
        print(f"Data loading pipeline failed: {e}")
        sys.exit(1)