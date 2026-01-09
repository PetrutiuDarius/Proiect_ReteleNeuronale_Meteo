# src/neural_network/evaluate.py
import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.neural_network.data_generator import TimeSeriesGenerator

def evaluate_model():
    print("==========================================")
    print("   STARTING MODEL EVALUATION (TEST SET)   ")
    print("==========================================")

    # Load paths
    model_path = os.path.join(config.BASE_DIR, 'models', 'trained_model.keras')
    test_data_path = os.path.join(config.DATA_DIR, 'test', 'test.csv')
    scaler_path = config.SCALER_PATH

    # Checks
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train_model.py first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Run split_data.py first.")

    # Load artifacts
    print("Loading model and scaler...")
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    print("Loading test data...")
    df_test = pd.read_csv(test_data_path)

    # Prepare data generator for test set
    # I use the same sliding window
    gen = TimeSeriesGenerator(
        input_width=config.SEQ_LENGTH,
        label_width=config.PREDICT_HORIZON,
        feature_cols=config.FEATURE_COLS,
        target_col=config.TARGET_COL
    )

    X_test, y_test = gen.create_sequences(df_test)
    print(f"Test samples generated: {X_test.shape[0]}")

    # Run interface (prediction)
    print("Running prediction on test set...")
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Denormalize data (inverse transform)
    # The model predicts scaled values (0-1) and we need real degrees Celsius.
    # The scaler expects 4 columns (temp, hum, pres, wind), but we only predicted Temp.
    # I need to create a dummy matrix to trick the scaler inverse_transform.

    def denormalize(y_scaled_array):
        # Create a placeholder matrix with zeros for other features
        dummy_matrix = np.zeros((len(y_scaled_array), len(config.FEATURE_COLS)))
        # Put our predicted temperature in the first column (index 0) - assuming Temp is the first feature
        dummy_matrix[:, 0] = y_scaled_array.flatten()
        # Inverse transform
        inversed = scaler.inverse_transform(dummy_matrix)
        # Returns only the temperature column
        return inversed[:,0]

    y_pred_real = denormalize(y_pred_scaled)
    y_true_real = denormalize(y_test)

    # Calculate metrics
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    r2 = r2_score(y_true_real, y_pred_real)

    print("\n------------------------------------------------")
    print(f"   RESULTS (Test Set 2024)")
    print("------------------------------------------------")
    print(f"   MAE  (Eroare Medie Absoluta): {mae:.4f} °C")
    print(f"   RMSE (Eroare Patratica Medie): {rmse:.4f} °C")
    print(f"   R2 Score (Potrivire): {r2:.4f}")
    print("------------------------------------------------")

    # Save metrics to JSON
    metrics = {
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2)
    }

    results_dir = os.path.join(config.BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to results/test_metrics.json")

    # Generate comparison plot (actual vs. predicted)
    print("Generating prediction plot...")
    plt.figure(figsize=(12,6))
    # I plot only the first 200 hours for clarity
    limit = 200
    plt.plot(y_true_real[:limit], label='Valori reale (actual)', color='blue')
    plt.plot(y_pred_real[:limit], label='Predicție AI (Predicted)', color='red', linestyle='--')

    plt.title(f'Prognoza meteo: Real vs. AI (Primele {limit} ore din test)')
    plt.xlabel('Timp (ore)')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(config.BASE_DIR, 'docs', 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f"Plor saved to {plot_path}")

if __name__ == "__main__":
    evaluate_model()