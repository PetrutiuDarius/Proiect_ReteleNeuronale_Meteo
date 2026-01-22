# src/preprocessing/split_data.py
"""
Data splitting & Normalization module.

This module implements the strategy for partitioning the hybrid dataset into
Training, Validation, and Testing sets while strictly adhering to time-series
validation principles (avoiding look-ahead bias).

Key responsibilities:
1. Chronological splitting: Ensures future data is not used to train past predictions.
2. Synthetic integration: Injects 'Black Swan' events only into the Training set.
3. Feature scaling: Fits normalization parameters strictly on Training data to prevent Data Leakage.
"""

import pandas as pd
import os
import sys
import joblib
from sklearn.preprocessing import MinMaxScaler
from src import config

def split_and_normalize_dataset() -> None:
    """
    Orchestrates the splitting and normalization pipeline.

    Architecture:
    - Train Set: 2020-2023 (Real) + All Synthetic Data (Extreme Events).
    - Validation Set: 2024 (Odd Months).
    - Test Set: 2024 (Even Months).

    The scaler handles all 9 inputs defined in config.FEATURE_COLS,
    ensuring Time Embeddings (Sin/Cos) are scaled alongside physical parameters.
    """
    print("Starting data splitting & normalization pipeline...")

    # Load the hybrid dataset
    if not os.path.exists(config.HYBRID_DATA_PATH):
        print(f"Hybrid dataset missing at: {config.HYBRID_DATA_PATH}")
        print("Execute 'main.py' (Phase 1 & 2) first.")
        sys.exit(1)

    try:
        df_full = pd.read_csv(config.HYBRID_DATA_PATH)

        # Handle cases where the index was saved without a name (resulting in 'Unnamed: 0')
        if 'timestamp' not in df_full.columns:
            if 'Unnamed: 0' in df_full.columns:
                print("[WARN] detected unnamed index column. Renaming to 'timestamp'.")
                df_full.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
            else:
                # If the timestamp is missing and no unnamed column, it might be the index already
                # but read_csv usually creates a RangeIndex if not specified.
                print(f"[CRITICAL] Column 'timestamp' not found. Columns are: {df_full.columns.tolist()}")
                sys.exit(1)

    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)

    # Convert timestamp to datetime objects for temporal filtering
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], errors='coerce')

    # I set timestamp as an index for easy slicing but keep it as a column for debugging
    df_full.set_index('timestamp', inplace=True, drop=False)

    # Validation: Ensure all required parameters exist
    missing_cols = [col for col in config.FEATURE_COLS if col not in df_full.columns]
    if missing_cols:
        print(f"Dataset is missing features required by config: {missing_cols}")
        sys.exit(1)

    # Segregate real vs. Synthetic data
    # 'is_simulated' flag allows us to treat real history and synthetic extremes differently
    df_real = df_full[df_full['is_simulated'] == 0].copy()
    df_synthetic = df_full[df_full['is_simulated'] == 1].copy()

    print(f"Real samples: {len(df_real)} | Synthetic samples: {len(df_synthetic)}")

    # Chronological splitting strategy
    # I split 2024 into Val/Test to simulate "current year" performance.
    # Alternating months ensure we cover all seasons in both Val and Test.

    train_real = df_real[df_real.index.year < 2024].copy()
    df_2024 = df_real[df_real.index.year == 2024].copy()

    val_mask = (df_2024.index.month % 2 != 0)  # Odd months (Jan, Mar, ...)
    test_mask = (df_2024.index.month % 2 == 0)  # Even months (Feb, Apr, ...)

    val_real = df_2024[val_mask].copy()
    test_real = df_2024[test_mask].copy()

    # Construct the final datasets
    # Synthetic data is added ONLY to training.
    # I do not want to validate/test on fake data; I want to benchmark against reality.
    train_final = pd.concat([train_real, df_synthetic])
    val_final = val_real
    test_final = test_real

    # 6. Normalization (MinMax Scaling)
    # Neural Networks converge faster with features in range [0, 1].
    # I fit the scaler ONLY on train_final to avoid data leakage.

    print("Fitting Scaler on training data (real + synthetic)...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = config.FEATURE_COLS

    # Fit on training data
    scaler.fit(train_final[feature_cols])

    # Transform all partitions
    # Values in test might go slightly outside [0,1] if they exceed training extremes.
    # This is expected behavior in production (the real world is unpredictable).
    train_final[feature_cols] = scaler.transform(train_final[feature_cols])
    val_final[feature_cols] = scaler.transform(val_final[feature_cols])
    test_final[feature_cols] = scaler.transform(test_final[feature_cols])

    # Save artifacts
    # Ensuring directory structures exist
    for path in [os.path.join(config.DATA_DIR, sub) for sub in ['train', 'validation', 'test']]:
        os.makedirs(path, exist_ok=True)

    # Save processed CSVs
    train_final.to_csv(os.path.join(config.DATA_DIR, 'train', 'train.csv'), index=False)
    val_final.to_csv(os.path.join(config.DATA_DIR, 'validation', 'validation.csv'), index=False)
    test_final.to_csv(os.path.join(config.DATA_DIR, 'test', 'test.csv'), index=False)

    # Save the scaler model
    os.makedirs(os.path.dirname(config.SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)

    # Final report
    print("\n[SUCCESS] Data splitting & normalization complete.")
    print("-" * 60)
    print(f"{'Set':<15} | {'Composition':<30} | {'Count':<10}")
    print("-" * 60)
    print(f"{'Train':<15} | {'Real(20-23) + Synthetic':<30} | {len(train_final):<10}")
    print(f"{'Validation':<15} | {'Real(2024 Odd Months)':<30} | {len(val_final):<10}")
    print(f"{'Test':<15} | {'Real(2024 Even Months)':<30} | {len(test_final):<10}")
    print(f"{'Scaler':<15} | {'Saved to':<30} | {config.SCALER_PATH}")
    print("-" * 60)

if __name__ == "__main__":
    split_and_normalize_dataset()