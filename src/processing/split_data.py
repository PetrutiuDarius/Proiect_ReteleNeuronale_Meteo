# src/processing/split_data.py
import pandas as pd
import numpy as np
import os
import sys
import joblib  # For saving the Scaler
from sklearn.preprocessing import MinMaxScaler
from src import config


def split_and_normalize_dataset() -> None:
    """
    Core processing pipeline:
    1. Loads the Hybrid Dataset (Real + Synthetic).
    2. Splits Real data into Train (2020-2023) and Val/Test (2024).
    3. Adds ALL Synthetic data to the Training set (to learn extremes).
    4. Normalizes data using a Scaler fitted ONLY on Training data (to prevent data leakage).
    5. Saves the final artifacts.
    """
    print("[INFO] Starting Dataset Splitting & Normalization...")

    # 1. Load Hybrid Data
    if not os.path.exists(config.HYBRID_DATA_PATH):
        print(f"[ERROR] Hybrid dataset not found at: {config.HYBRID_DATA_PATH}")
        print("[HINT] Run 'main.py' step 1 & 2 first.")
        sys.exit(1)

    df_full = pd.read_csv(config.HYBRID_DATA_PATH)

    # Ensure timestamp is datetime (errors='coerce' handles distinct synthetic timestamps if any)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')

    # Set index but keep it as column too for filtering
    df_full.set_index('timestamp', inplace=True, drop=False)

    # 2. Separate Real vs Synthetic
    # We rely on the flag we created in synthetic_generator.py
    df_real = df_full[df_full['is_simulated'] == 0].copy()
    df_synthetic = df_full[df_full['is_simulated'] == 1].copy()

    print(f"[DEBUG] Real samples: {len(df_real)} | Synthetic samples: {len(df_synthetic)}")

    # 3. Split Real Data Logic
    # Train: < 2024
    # Val/Test: == 2024 (Alternating months)

    train_real = df_real[df_real.index.year < 2024].copy()
    df_2024 = df_real[df_real.index.year == 2024].copy()

    val_mask = (df_2024.index.month % 2 != 0)  # Odd months
    test_mask = (df_2024.index.month % 2 == 0)  # Even months

    val_real = df_2024[val_mask].copy()
    test_real = df_2024[test_mask].copy()

    # 4. Construct Final Sets
    # Training = Real History (2020-2023) + All Synthetic Extremes
    # We concatenate them. The Neural Network Data Generator (later) will handle the jump
    # between them so we don't feed a window spanning across real/synthetic boundary.
    train_final = pd.concat([train_real, df_synthetic])

    # Validation & Test remain purely Real Data to benchmark reality
    val_final = val_real
    test_final = test_real

    # 5. Normalization (CRITICAL STEP)
    # We must fit the scaler ONLY on the Training set.
    # If we fit on Test, the model "peeks" into the future (Data Leakage).

    print("[INFO] Fitting Scaler on Training Data (Real + Synthetic)...")
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Columns to scale (exclude timestamp and flag)
    feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']

    # Fit
    scaler.fit(train_final[feature_cols])

    # Transform all sets
    train_final[feature_cols] = scaler.transform(train_final[feature_cols])
    val_final[feature_cols] = scaler.transform(val_final[feature_cols])
    test_final[feature_cols] = scaler.transform(test_final[feature_cols])

    # 6. Save Artifacts
    os.makedirs(os.path.join(config.DATA_DIR, 'train'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'test'), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'scalers'), exist_ok=True)

    train_final.to_csv(os.path.join(config.DATA_DIR, 'train', 'train.csv'), index=False)
    val_final.to_csv(os.path.join(config.DATA_DIR, 'validation', 'validation.csv'), index=False)
    test_final.to_csv(os.path.join(config.DATA_DIR, 'test', 'test.csv'), index=False)

    # Save the scaler object itself to use it later for inverse_transform (denormalization)
    scaler_path = os.path.join(config.DATA_DIR, 'scalers', 'minmax_scaler.pkl')
    joblib.dump(scaler, scaler_path)

    print("\n[SUCCESS] Data Splitting & Normalization Complete.")
    print("-" * 60)
    print(f"{'Set':<15} | {'Composition':<30} | {'Count':<10}")
    print("-" * 60)
    print(f"{'Train':<15} | {'Real(20-23) + Synthetic':<30} | {len(train_final):<10}")
    print(f"{'Validation':<15} | {'Real(2024 Odd Months)':<30} | {len(val_final):<10}")
    print(f"{'Test':<15} | {'Real(2024 Even Months)':<30} | {len(test_final):<10}")
    print(f"{'Scaler':<15} | {'Saved for later use':<30} | {scaler_path}")
    print("-" * 60)


if __name__ == "__main__":
    split_and_normalize_dataset()