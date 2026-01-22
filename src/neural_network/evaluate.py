# src/neural_network/evaluate.py
"""
Model Evaluation Module.

Loads the trained model and test data, performs multi-target inference,
calculates regression metrics (MAE, RMSE, R2) for each weather parameter individually,
and generates comparative plots.
"""

import os
import warnings
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Environment Setup (Clean Console) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

from src import config
from src.neural_network.data_generator import TimeSeriesGenerator


def apply_physics_constraints(data: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Post-preprocessing filter to enforce physical laws on predictions.

    Rules applied:
    1. Non-Negativity: Variables like Rain, Humidity, Wind cannot be negative.
    2. Saturation: Humidity cannot exceed 100%.
    3. Noise gating: Very small precipitation values (<0.05mm) are clamped to 0.
    """
    corrected_data = data.copy()

    for i, name in enumerate(feature_names):
        # Clip negative values to 0 for scalar physical quantities
        if name in ['humidity', 'wind_speed', 'precipitation']:
            corrected_data[:, i] = np.maximum(corrected_data[:, i], 0.0)

        # Cap humidity at 100%
        if name == 'humidity':
            corrected_data[:, i] = np.minimum(corrected_data[:, i], 100.0)

        # Rain noise gate (suppress 'micro-rain' hallucinations)
        if name == 'precipitation':
            # Values smaller than 0.05mm are likely model noise, treat as dry
            corrected_data[:, i] = np.where(corrected_data[:, i] < 0.05, 0.0, corrected_data[:, i])

    return corrected_data

def evaluate_model():
    print("==========================================")
    print("   STARTING MODEL EVALUATION (TEST SET)   ")
    print("==========================================")

    # Load paths
    test_data_path = os.path.join(config.DATA_DIR, 'test', 'test.csv')

    # Checks
    if not os.path.exists(config.MODEL_PATH):
        print(f"Model not found at {config.MODEL_PATH}. Run training first.")
        return
    if not os.path.exists(config.SCALER_PATH):
        print(f"Scaler not found at {config.SCALER_PATH}. Run preprocessing first.")
        return

    # Load artifacts
    print("Loading model and scaler...")
    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    print("Loading test data...")
    df_test = pd.read_csv(test_data_path)

    # Prepare data generator for test set
    # I use the same sliding window
    gen = TimeSeriesGenerator(
        input_width=config.SEQ_LENGTH,
        label_width=config.PREDICT_HORIZON,
        feature_cols=config.FEATURE_COLS,
        target_cols=config.TARGET_COLS
    )

    X_test, y_test = gen.create_sequences(df_test)
    print(f"Test samples generated: {X_test.shape}")

    # Run interface (prediction)
    print("Running prediction on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Denormalization & Post-Processing
    print("Denormalize data...")

    # Advanced Denormalization Logic for Asymmetric I/O
    # I create a dummy matrix matching the scaler's expected shape (9 cols)
    # and fill the target columns, then extract them back.

    def denormalize_targets(pred_array):
        # Create placeholder with correct shape (N, 9)
        dummy = np.zeros((pred_array.shape[0], len(config.FEATURE_COLS)))
        # Fill the physical columns (first 5)
        physical_idx = len(config.TARGET_COLS)
        dummy[:, :physical_idx] = pred_array
        # Inverse transform
        inversed = scaler.inverse_transform(dummy)
        # Return only physical columns
        return inversed[:, :physical_idx]

    # Check if Scaler expects 9 cols but we have 5
    if scaler.n_features_in_ != y_pred_scaled.shape[1]:
        y_pred_real = denormalize_targets(y_pred_scaled)
        y_true_real = denormalize_targets(y_test)
    else:
        # Fallback if shapes match
        y_pred_real = scaler.inverse_transform(y_pred_scaled)
        y_true_real = scaler.inverse_transform(y_test)

    # Apply physics constraints
    print("Applying physics-informed post-preprocessing...")
    y_pred_real = apply_physics_constraints(y_pred_real, config.TARGET_COLS)

    # Metrics and plotting loop
    metrics_json = {}
    feature_names = config.TARGET_COLS

    # Initialize plot (5 sublots, one for each feature)
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(12, 20), sharex=True)
    limit = 200 # I plot only the first 200 hours

    print("\n" + "="*50)
    print("   DETAILED PERFORMANCE REPORT")
    print("=" * 50)

    for i, col_name in enumerate(feature_names):
        # Extract data for this specific feature
        true_vals = y_true_real[:, i]
        pred_vals = y_pred_real[:, i]

        # Calculate metrics
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        r2 = r2_score(true_vals, pred_vals)

        # Save to dict
        metrics_json[f"{col_name}_mae"] = float(mae)
        metrics_json[f"{col_name}_rmse"] = float(rmse)
        metrics_json[f"{col_name}_r2"] = float(r2)

        print(f"\n   PARAMETER: {col_name.upper()}")
        print("-" * 48)
        print(f"   RESULTS (Test Set 2024 - {col_name})")
        print("-" * 48)
        print(f"   MAE  (Eroare Medie Absoluta): {mae:.4f}")
        print(f"   RMSE (Eroare Patratica Medie): {rmse:.4f}")
        print(f"   R2 Score (Potrivire): {r2:.4f}")
        print("-" * 48)

        # Plotting
        ax = axes[i]
        ax.plot(true_vals[:limit], label='Real (Actual)', color='blue')
        ax.plot(pred_vals[:limit], label='AI (Predicted)', color='red', linestyle='--')

        if col_name == 'precipitation':
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

        ax.set_ylabel(f'{col_name}')
        ax.set_title(f'Prognoza: {col_name.capitalize()}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Finalize plot
    axes[-1].set_xlabel('Timp (Ore)')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(config.BASE_DIR, 'docs', 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f"\nCombined plot saved to: {plot_path}")

    # Save JSON metrics
    results_dir = os.path.join(config.BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f"Metrics saved to results/test_metrics.json")

if __name__ == "__main__":
    evaluate_model()