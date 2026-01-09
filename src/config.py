# src/config.py
import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'weather_history_raw.csv')
GENERATED_DATA_PATH = os.path.join(DATA_DIR, 'generated', 'synthetic_extremes.csv')
HYBRID_DATA_PATH = os.path.join(DATA_DIR, 'generated', 'hybrid_dataset.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'final_normalized.csv')

# --- LOCATION SETTINGS (Default: Bucharest) ---
LOCATION = {
    "name": "Bucharest",
    "lat": 44.4323,
    "lon": 26.1063,
    "timezone": "Europe/Bucharest"
}

# --- API SETTINGS ---
# Fetching 5 full years (2020-2024)
API_START_DATE = "2020-01-01"
API_END_DATE = "2024-12-31"
API_URL_TEMPLATE = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude={lat}&longitude={lon}&"
    "start_date={start}&end_date={end}&"
    "hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&"
    "timezone={timezone}&format=csv&wind_speed_unit=ms"
)

# --- SYNTHETIC GENERATION SETTINGS ---
# Definitions for "Black Swan" events specific to the location's climate
EXTREME_SCENARIOS = {
    # 1. Heatwave: Temp > historical max (e.g., >42°C in Bucharest)
    "heatwave": {
        "temp_min": 40.0,
        "temp_max": 44.0,
        "duration_hours": (48, 120),  # 2 to 5 days
        "occurrence_prob": 0.05       # Probability of occurring in summer
    },
    # 2. Severe Storm: High wind, sudden pressure drop
    "storm": {
        "wind_speed_min": 20.0,       # m/s (approx 72 km/h)
        "wind_speed_max": 30.0,       # m/s (approx 108 km/h)
        "pressure_drop": 15.0,        # hPa drop
        "duration_hours": (2, 6),
        "occurrence_prob": 0.03
    },
    # 3. Late Frost: Negative temps in April/May
    "late_frost": {
        "temp_min": -3.0,
        "temp_max": 0.0,
        "months": [4, 5],             # April, May
        "duration_hours": (4, 10),    # Overnight
        "occurrence_prob": 0.10
    }
}

# Target size for the synthetic dataset (approx 25k hours)
SYNTHETIC_SAMPLES_TARGET = 25000

# --- NEURAL NETWORK HYPERPARAMETERS ---
# How many past hours the model sees to make a prediction
SEQ_LENGTH = 24

# How many hours into the future we want to predict
PREDICT_HORIZON = 6

# Features used for training (must match dataframe columns)
# Note: 'is_simulated' is excluded from input features to force the model to learn patterns, not labels
FEATURE_COLS = ['temperature', 'humidity', 'pressure', 'wind_speed']
TARGET_COL = 'temperature' # We are predicting temperature

# Training settings
BATCH_SIZE = 64        # Number of samples processed before updating weights
EPOCHS = 50            # NUmber of complete passes through the dataset
LEARNING_RATE = 0.001  # Step size for the optimizer
PATIENCE = 10          # Early stopping patience (stop if no improvement after 10 epochs)
