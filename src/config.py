# src/config.py
"""
Global Configuration Module.

This file serves as the Single Source of Truth (SSOT) for the entire application.
It defines directory paths, API endpoints, physical constants, and Neural Network
hyperparameters.

Modifying parameters here automatically propagates changes to Data Acquisition,
Processing, Training, and Inference modules.
"""

import os

# =============================================================================
#  PROJECT STRUCTURE & PATHS
# =============================================================================
# Resolve the project root dynamically to ensure portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# File Artifacts
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'weather_history_raw.csv')
GENERATED_DATA_PATH = os.path.join(DATA_DIR, 'generated', 'synthetic_extremes.csv')
HYBRID_DATA_PATH = os.path.join(DATA_DIR, 'generated', 'hybrid_dataset.csv')
# Note: Processed data is split into train/val/test CSVs in their respective folders

# Model Artifacts
SCALER_PATH = os.path.join(CONFIG_DIR, 'preprocessing_params.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.keras')

# =============================================================================
#  LOCATION & API SETTINGS
# =============================================================================
LOCATION = {
    "name": "Bucharest",
    "lat": 44.4323,
    "lon": 26.1063,
    "timezone": "Europe/Bucharest"
}

# Fetching 5 full years (2020-2024)
API_START_DATE = "2020-01-01"
API_END_DATE = "2024-12-31"

# Open-Meteo API Template
# I explicitly request 5 parameters: Temp, Humidity, Pressure, Wind, Precipitation
API_URL_TEMPLATE = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude={lat}&longitude={lon}&"
    "start_date={start}&end_date={end}&"
    "hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation&"
    "timezone={timezone}&format=csv&wind_speed_unit=ms"
)

# =============================================================================
#  SYNTHETIC DATA GENERATION (Black Swan Events)
# =============================================================================
# Constraints used to generate physically plausible extreme weather events
EXTREME_SCENARIOS = {
    "heatwave": {
        "temp_min": 40.0,
        "temp_max": 44.0,
        "duration_hours": (48, 120),
        "occurrence_prob": 0.05
    },
    "storm": {
        "wind_speed_min": 20.0,
        "wind_speed_max": 30.0,
        "pressure_drop": 15.0,
        "duration_hours": (2, 6),
        "occurrence_prob": 0.03
    },
    "late_frost": {
        "temp_min": -3.0,
        "temp_max": 0.0,
        "months": [4, 5],
        "duration_hours": (4, 10),
        "occurrence_prob": 0.10
    }
}

# Target size for the synthetic dataset (approx 25k hours)
SYNTHETIC_SAMPLES_TARGET = 25000

# =============================================================================
#  NEURAL NETWORK ARCHITECTURE
# =============================================================================
# Sliding Window: 24h allows capturing the full diurnal cycle (Day/Night patterns)
SEQ_LENGTH = 24

# Prediction Horizon: I predict the NEXT hour (t+1).
# Multistep forecasting (e.g., 24h ahead) is handled via autoregression in the UI.
PREDICT_HORIZON = 1

# Input Features (9 Total)
# Includes 5 Physical parameters + 4 Cyclical Time Embeddings
FEATURE_COLS = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed',
    'precipitation',
    'day_sin', 'day_cos',   # Time of Day (Circadian rhythm)
    'year_sin', 'year_cos'  # Seasonality (Annual rhythm)
]

# Output Targets (5 Total)
# I only predict the physical state of the atmosphere
TARGET_COLS = [
    'temperature',
    'humidity',
    'pressure',
    'wind_speed',
    'precipitation'
]

# =============================================================================
#  TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 64        # Number of samples processed before updating weights
EPOCHS = 50            # NUmber of complete passes through the dataset
LEARNING_RATE = 0.001  # Step size for the optimizer
PATIENCE = 5           # Early stopping patience (stop if no improvement after 5 epochs)


# =============================================================================
#  LOGIC CONSTANTS
# =============================================================================
# Threshold to distinguish Rain vs Snow in the UI/Logic layer
# If Precip > 0 and Temp <= SNOW_TEMP_THRESHOLD, we classify as SNOW.
SNOW_TEMP_THRESHOLD = 0.5
