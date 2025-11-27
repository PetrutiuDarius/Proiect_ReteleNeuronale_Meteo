# src/data_loader.py
import pandas as pd
import requests
import os
import sys
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
# Open-Meteo URL: Fetching data from 2020 to end of 2024
# We explicitly request wind speed in m/s to match our sensor requirements.
DATA_URL = "https://archive-api.open-meteo.com/v1/archive?latitude=44.4323&longitude=26.1063&start_date=2020-01-01&end_date=2024-12-31&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&timezone=Europe%2FBucharest&format=csv&wind_speed_unit=ms"

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bucharest_weather_raw.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bucharest_normalized.csv')


def download_data() -> None:
    """
    Downloads raw weather data from Open-Meteo API if it doesn't exist locally.
    """
    if os.path.exists(RAW_DATA_PATH):
        print(f"[INFO] Raw data file found at: {RAW_DATA_PATH}")
        print("[INFO] Skipping download.")
        return

    print(f"[INFO] Downloading weather data for Bucharest (2020-2024)...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()  # Check for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download data: {e}")
        sys.exit(1)

    # Ensure directory exists
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    with open(RAW_DATA_PATH, 'wb') as f:
        f.write(response.content)

    print(f"[SUCCESS] Raw data saved successfully.")


def process_data() -> None:
    """
    Loads raw data, cleans column names, interpolates missing values,
    normalizes features (0-1 range), and saves the processed dataset.
    """
    print("[INFO] Starting data processing pipeline...")

    # 1. Load Data
    try:
        # Skip the first 2 rows containing metadata headers from Open-Meteo
        df = pd.read_csv(RAW_DATA_PATH, skiprows=2)
    except FileNotFoundError:
        print(f"[ERROR] Raw file not found at {RAW_DATA_PATH}.")
        print("[HINT] Please run the download step first.")
        sys.exit(1)

    # 2. Rename Columns for consistency
    rename_map = {
        'time': 'timestamp',
        'temperature_2m (°C)': 'temperature',
        'relative_humidity_2m (%)': 'humidity',
        'surface_pressure (hPa)': 'pressure',
        'wind_speed_10m (m/s)': 'wind_speed'
    }
    df.rename(columns=rename_map, inplace=True)

    # 3. Validate Columns
    required_cols = ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"[ERROR] Critical columns missing: {missing_cols}")
        print(f"[DEBUG] Available columns: {df.columns.tolist()}")
        print("[HINT] Delete the file in 'data/raw/' and restart to re-download.")
        sys.exit(1)

    # 4. Data Cleaning & Indexing
    df = df[required_cols]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 5. Handle Missing Values (Time-based interpolation)
    # Weather doesn't jump instantly, so linear interpolation is safe.
    df.interpolate(method='time', inplace=True)

    # 6. Normalization (MinMax Scaling)
    # Neural Networks converge faster with data in [0, 1] range.
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # 7. Save Processed Data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_normalized.to_csv(PROCESSED_DATA_PATH)

    print(f"[SUCCESS] Data processed and saved to: {PROCESSED_DATA_PATH}")
    print(f"[INFO] Dataset size: {df_normalized.shape[0]} hours (rows)")
    print("-" * 30)
    print("Sample Data (First 5 rows):")
    print(df_normalized.head())
    print("-" * 30)