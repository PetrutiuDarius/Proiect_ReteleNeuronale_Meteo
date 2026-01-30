# src/neural_network/evaluate.py
"""
Model Evaluation & Inference Module.

This script is responsible for the rigorous assessment of the trained Neural Network.
It loads the model artifacts, performs inference on the unseen Test Set (2024),
and computes standard regression metrics.

Key Features:
- Physics-Informed Post-Processing: Applies domain constraints (e.g., non-negative rain).
- Asymmetric Loss Support: Registers custom loss functions for model loading.
- Advanced Denormalization: Handles shape mismatches between scaler inputs and model outputs.
- Visualization: Generates comparative time-series plots for qualitative analysis.
"""

import os
import warnings
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Environment Setup (Clean Console) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

from src import config
from src.neural_network.data_generator import TimeSeriesGenerator


# -------------------------------------------------------------------------
# CUSTOM OBJECT REGISTRATION
# -------------------------------------------------------------------------
# We must register this function so Keras can reconstruct the model graph
# from the saved file without throwing a DeserializationError.
@tf.keras.utils.register_keras_serializable()
def asymmetric_precipitation_loss(y_true, y_pred):
    """
    Re-definition of the custom loss function used during training.
    Strictly required for the `load_model` process.
    """
    squared_error = tf.square(y_true - y_pred)
    overestimation_mask = tf.cast(tf.greater(y_pred, y_true), tf.float32)

    # Target specific column: Precipitation (Index 4)
    rain_col_idx = 4
    feature_count = 5

    # Create broadcast mask
    rain_column_mask = tf.one_hot(indices=[rain_col_idx] * tf.shape(y_true)[0], depth=feature_count)
    rain_column_mask = tf.reshape(rain_column_mask, tf.shape(y_true))

    # Apply penalty for false positives
    penalty_magnitude = 20.0
    penalty_factor = 1.0 + (overestimation_mask * rain_column_mask * penalty_magnitude)

    return tf.reduce_mean(squared_error * penalty_factor)


# -------------------------------------------------------------------------
# POST-PROCESSING LOGIC
# -------------------------------------------------------------------------
def apply_physics_constraints(data: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Applies domain-specific physical constraints to the raw model predictions.
    Deep Learning models are purely mathematical and may produce physically
    impossible values (e.g., negative rain). This filter corrects them.

    Applied Rules:
    1. Non-Negativity: Scalar quantities (Wind, Rain, Humidity) cannot be < 0.
    2. Saturation: Relative Humidity cannot exceed 100%.
    3. Noise Gating: Precipitations < 0.1mm are treated as sensor/model noise and clamped to 0.
    """
    corrected_data = data.copy()

    for i, name in enumerate(feature_names):
        # Rule 1: Clip negative values for physical scalars
        if name in ['humidity', 'wind_speed', 'precipitation']:
            corrected_data[:, i] = np.maximum(corrected_data[:, i], 0.0)

        # Rule 2: Cap humidity at saturation point
        if name == 'humidity':
            corrected_data[:, i] = np.minimum(corrected_data[:, i], 100.0)

        # Rule 3: Precipitation Noise Gate (The "Zero-Inflation" Fix)
        # This eliminates the "constant drizzle" artifact in regression models.
        if name == 'precipitation':
            threshold_mm = 0.1
            mask_noise = corrected_data[:, i] < threshold_mm
            corrected_data[mask_noise, i] = 0.0

            count_filtered = np.sum(mask_noise)
            if count_filtered > 0:
                print(f"   [Physics] Filtered {count_filtered} micro-rain events (<{threshold_mm}mm) to 0.0mm.")

    return corrected_data


def denormalize_targets(pred_array: np.ndarray, scaler) -> np.ndarray:
    """
    Handles dimension mismatch during inverse transformation.
    The Scaler expects 9 features (Inputs), but the Model outputs 5 features (Targets).
    We create a dummy matrix to satisfy the Scaler's API.
    """
    # Create a placeholder matrix (N_samples, 9_features)
    dummy = np.zeros((pred_array.shape[0], scaler.n_features_in_))

    # Fill the first 5 columns with our predictions
    # (Assuming targets are the first 5 columns in FEATURE_COLS)
    num_targets = pred_array.shape[1]
    dummy[:, :num_targets] = pred_array

    # Perform inverse transform
    inversed_matrix = scaler.inverse_transform(dummy)

    # Extract only the relevant columns back
    return inversed_matrix[:, :num_targets]


# -------------------------------------------------------------------------
# MAIN EVALUATION PIPELINE
# -------------------------------------------------------------------------
def evaluate_model():
    print("==========================================")
    print("   STARTING MODEL EVALUATION (TEST SET)   ")
    print("==========================================")

    # 1. Prerequisite Checks
    test_data_path = os.path.join(config.DATA_DIR, 'test', 'test.csv')

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model artifact missing: {config.MODEL_PATH}")
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(f"Scaler artifact missing: {config.SCALER_PATH}")

    # 2. Load Artifacts
    print("Loading model and scaler...")
    # We pass custom_objects explicitly to ensure robust loading
    try:
        model = load_model(config.MODEL_PATH, custom_objects={
            'asymmetric_precipitation_loss': asymmetric_precipitation_loss
        })
    except Exception as e:
        print(f"[CRITICAL] Model loading failed. Error: {e}")
        return

    scaler = joblib.load(config.SCALER_PATH)

    # 3. Load & Prepare Data
    print(f"Loading test data from {test_data_path}...")
    df_test = pd.read_csv(test_data_path)

    print("Initializing test generator...")
    gen = TimeSeriesGenerator(
        input_width=config.SEQ_LENGTH,
        label_width=config.PREDICT_HORIZON,
        feature_cols=config.FEATURE_COLS,
        target_cols=config.TARGET_COLS
    )

    X_test, y_test = gen.create_sequences(df_test)
    print(f"   -> Inference Batch Size: {X_test.shape[0]} samples")

    # 4. Run Inference
    print("Running inference on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # 5. Denormalization & Post-Processing
    print("Denormalizing and applying physics constraints...")

    # Handle scaler dimension mismatch safely
    if scaler.n_features_in_ != y_pred_scaled.shape[1]:
        y_pred_real = denormalize_targets(y_pred_scaled, scaler)
        y_true_real = denormalize_targets(y_test, scaler)
    else:
        y_pred_real = scaler.inverse_transform(y_pred_scaled)
        y_true_real = scaler.inverse_transform(y_test)

    # Apply the noise gate and non-negativity rules
    y_pred_final = apply_physics_constraints(y_pred_real, config.TARGET_COLS)

    # 6. Metrics Calculation & Reporting
    metrics_json = {}
    feature_names = config.TARGET_COLS

    # Setup visualization
    limit = 1000  # Plot first 500 hours (~21 days)
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 20), sharex=True)

    print("\n" + "=" * 50)
    print("   DETAILED PERFORMANCE REPORT (2024 Data)")
    print("=" * 50)

    for i, col_name in enumerate(feature_names):
        # Slicing data for specific feature
        truth = y_true_real[:, i]
        pred = y_pred_final[:, i]

        # Core Metrics
        mae = mean_absolute_error(truth, pred)
        rmse = np.sqrt(mean_squared_error(truth, pred))
        r2 = r2_score(truth, pred)

        # Log metrics
        metrics_json[f"{col_name}_mae"] = float(mae)
        metrics_json[f"{col_name}_rmse"] = float(rmse)
        metrics_json[f"{col_name}_r2"] = float(r2)

        # Print Report
        print(f"\n   PARAMETER: {col_name.upper()}")
        print("-" * 48)
        print(f"   MAE  (Mean Absolute Error):  {mae:.4f}")
        print(f"   RMSE (Root Mean Squared):    {rmse:.4f}")
        print(f"   R2   (Coefficient of Det.):  {r2:.4f}")

        # Plotting Subplot
        ax = axes[i]
        ax.plot(truth[:limit], label='Real (Ground Truth)', color='blue', linewidth=1.5)
        ax.plot(pred[:limit], label='AI Prediction', color='red', linestyle='--', linewidth=1.5)

        # Add zero-line for precipitation for clarity
        if col_name == 'precipitation':
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        ax.set_ylabel(f'{col_name}')
        ax.set_title(f'Forecast Analysis: {col_name.capitalize()}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # 7. Finalize Artifacts
    axes[-1].set_xlabel('Time Steps (Hours)')
    plt.tight_layout()

    # Save Plots
    plot_path = os.path.join(config.BASE_DIR, 'docs', 'prediction_plot.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"\n[OUTPUT] Comparative plot saved to: {plot_path}")

    # Save Metrics
    metrics_path = os.path.join(config.BASE_DIR, 'results', 'test_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f"[OUTPUT] Metrics exported to: {metrics_path}")


if __name__ == "__main__":
    evaluate_model()