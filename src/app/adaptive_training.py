# src/app/adaptive_training.py
"""
Adaptive AI Training Orchestrator.

This module provides the logic to dynamically retrain the LSTM model for a specific
geographic location. It leverages the existing modular architecture (data loading,
preprocessing, model definition) but executes the pipeline in a localized context
to avoid overwriting the thesis's main artifacts.

Key Features:
1.  **Dynamic Data Acquisition:** Fetches historical weather data for any (lat, lon).
2.  **Isolated Scaler:** Fits a new MinMaxScaler specific to the local climate.
3.  **Model Reuse:** Reuses the exact same LSTM topology defined in `src.neural_network.model`.
4.  **Hot-Swap Readiness:** Saves artifacts in a structure ready for dynamic loading by the Dashboard.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import Dict, Union, Callable, Optional

# --- PROJECT MODULE IMPORTS ---
# Ensure the project root is in the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src import config
from src.data_acquisition.data_loader import fetch_open_meteo_history
from src.neural_network.data_generator import TimeSeriesGenerator
from src.neural_network.model import build_lstm_model

# =============================================================================
#  CUSTOM LOSS FUNCTION
# =============================================================================
@tf.keras.utils.register_keras_serializable()
def asymmetric_precipitation_loss(y_true, y_pred):
    """
    Computes the Asymmetric Loss for precipitation forecasting.

    This function penalizes underestimation of rain events significantly more than
    overestimation. This strategy combats the class imbalance inherent in weather data
    (where >90% of hours are dry).

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.

    Returns:
        tf.Tensor: Scalar loss value.
    """
    squared_error = tf.square(y_true - y_pred)
    overestimation_mask = tf.cast(tf.greater(y_pred, y_true), tf.float32)

    # Dynamic identification of the precipitation column based on config
    rain_col_idx = 4
    feature_count = len(config.TARGET_COLS)

    # Create a mask that isolates the precipitation feature
    rain_column_mask = tf.one_hot(indices=[rain_col_idx] * tf.shape(y_true)[0], depth=feature_count)
    rain_column_mask = tf.reshape(rain_column_mask, tf.shape(y_true))

    # Apply penalty factor (Magnitude 20.0 determined via hyperparameter tuning)
    penalty_magnitude = 20.0
    penalty_factor = 1.0 + (overestimation_mask * rain_column_mask * penalty_magnitude)

    return tf.reduce_mean(squared_error * penalty_factor)

# =============================================================================
#  FEATURE ENGINEERING UTILS
# =============================================================================
def calculate_time_features(timestamp: pd.Timestamp) -> list:
    """
    Transforms linear time into cyclical features (Sin/Cos).

    This embedding allows the neural network to understand temporal proximity
    (e.g., 23:00 is close to 00:00, December is close to January).

    Args:
        timestamp (pd.Timestamp): The datetime object to process.

    Returns:
        list: [day_sin, day_cos, year_sin, year_cos]
    """
    day = 24 * 60 * 60
    year = 365.2425 * day
    ts_s = timestamp.timestamp()
    return [
        np.sin(ts_s * (2 * np.pi / day)),
        np.cos(ts_s * (2 * np.pi / day)),
        np.sin(ts_s * (2 * np.pi / year)),
        np.cos(ts_s * (2 * np.pi / year))
    ]

# =============================================================================
#  MAIN ORCHESTRATION FUNCTION
# =============================================================================
def train_adaptive_model(
        lat: float,
        lon: float,
        progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Union[str, float, int]]:
    """
    Executes the End-to-End pipeline for training a location-specific model.

    Pipeline Steps:
    1.  **Data Ingestion:** Fetch 5 years of historical data for (lat, lon).
    2.  **Preprocessing:** Clean data, apply Log-Transform, and generate Time Embeddings.
    3.  **Scaling:** Fit a new MinMaxScaler specific to the local climate distribution.
    4.  **Sequencing:** Create sliding windows (X, y) for LSTM input.
    5.  **Training:** Compile and fit the LSTM model using Early Stopping.
    6.  **Persistence:** Save model, scaler, and performance metrics to a dedicated folder.

    Args:
        lat (float): Latitude of the target location.
        lon (float): Longitude of the target location.
        progress_callback (func, optional): Function to report progress updates to UI.

    Returns:
        dict: Training metadata (MAE, Loss, Date) or error message.
    """

    # --- ENVIRONMENT SETUP ---
    # Define a unique directory for this location to avoid collisions
    model_dir = f"adaptive_models/{lat}_{lon}"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    metrics_path = os.path.join(model_dir, "metrics.json")

    # --- DATA ACQUISITION ---
    if progress_callback:
        progress_callback("Acquiring historical data (5 Years)...", 0.1)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

    try:
        # Reuse existing data loader logic but with dynamic coordinates
        df = fetch_open_meteo_history(lat, lon, start_date, end_date)
        if df is None or df.empty:
            return {"error": "Failed to download data from Open-Meteo API."}
    except Exception as e:
        return {"error": f"Data Loader Exception: {e}"}

    # --- PREPROCESSING & CLEANING ---
    if progress_callback:
        progress_callback("Preprocessing & Feature Engineering...", 0.3)

    # Handle missing values via linear interpolation (standard for time-series)
    df = df.interpolate(method='linear').ffill().bfill()

    # Apply Log-Transformation to Precipitation
    # This compresses the high dynamic range of rainfall data, stabilizing gradient descent.
    if 'precipitation' in df.columns:
        df['precipitation'] = np.log1p(df['precipitation'])

    # Generate Time Embeddings
    time_feats = [calculate_time_features(ts) for ts in df['timestamp']]
    time_df = pd.DataFrame(time_feats, columns=['day_sin', 'day_cos', 'year_sin', 'year_cos'])

    # Concatenate Features: Physics (5 cols) + Time (4 cols) = 9 Input Features
    try:
        data_physics = df[config.TARGET_COLS]
        data_full = pd.concat([data_physics.reset_index(drop=True), time_df.reset_index(drop=True)], axis=1)
    except KeyError as e:
        return {"error": f"Missing required columns in dataset: {e}"}

    # --- NORMALIZATION (SCALING) ---
    if progress_callback:
        progress_callback("Normalizing data (Fitting local Scaler)...", 0.4)

    # I must fit a new scaler because the min/max values (e.g., Temp in mountains vs sea)
    # will differ significantly from the generic Bucharest dataset.
    scaler = MinMaxScaler()
    data_scaled_numpy = scaler.fit_transform(data_full)

    # Persist the scaler for later inference
    joblib.dump(scaler, scaler_path)

    # Reconstruct DataFrame because TimeSeriesGenerator expects pandas input
    df_scaled = pd.DataFrame(data_scaled_numpy, columns=config.FEATURE_COLS)

    # --- SEQUENCE GENERATION ---
    if progress_callback:
        progress_callback("Generating LSTM Sequences...", 0.5)

    # Reuse the project's standardized Sequence Generator
    gen = TimeSeriesGenerator(
        input_width=config.SEQ_LENGTH,
        label_width=config.PREDICT_HORIZON,
        feature_cols=config.FEATURE_COLS,
        target_cols=config.TARGET_COLS
    )

    X, y = gen.create_sequences(df_scaled)

    if len(X) == 0:
        return {"error": "Insufficient data to generate sequences."}

    # Validation Split (90/10) - I keep validation data in memory (no disk IO needed)
    split_idx = int(len(X) * 0.9)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # --- MODEL TRAINING ---
    if progress_callback:
        progress_callback("Training Neural Network (Adapting weights)...", 0.6)

    # Construct the model architecture defined in src.neural_network.model
    input_shape = (config.SEQ_LENGTH, len(config.FEATURE_COLS))
    model = build_lstm_model(input_shape, learning_rate=0.001, output_units=len(config.TARGET_COLS))

    # IMPORTANT: Recompile with Custom Loss
    # The default build function uses MSE. I override it here to use the Asymmetric Loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=asymmetric_precipitation_loss,
        metrics=['mae']
    )

    # Early Stopping prevents overfitting and saves time
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,  # Sufficient for adaptation (Transfer Learning concept)
        batch_size=32,
        callbacks=[early_stop],
        verbose=0  # Silent mode to keep UI clean
    )

    # --- EVALUATION & SAVING ---
    if progress_callback:
        progress_callback("Finalizing and Saving artifacts...", 0.9)

    model.save(model_path)
    loss, mae = model.evaluate(X_val, y_val, verbose=0)

    metrics = {
        "location": f"{lat}, {lon}",
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mae": float(mae),
        "loss": float(loss),
        "data_points": len(df),
        "epochs_run": len(history.history['loss'])
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    if progress_callback:
        progress_callback("Process Complete!", 1.0)

    return metrics

# --- UNIT TEST ---
if __name__ == "__main__":
    # Simple test harness to verify functionality without running the full dashboard
    print(">>> Testing Adaptive Training Module...")

    # Callback simulator
    def print_progress(msg, val):
        print(f"[{val * 100:.0f}%] {msg}")

    # Test with Bucharest coordinates
    result = train_adaptive_model(44.43, 26.10, progress_callback=print_progress)
    print("\n>>> Result:", json.dumps(result, indent=2))