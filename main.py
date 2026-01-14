# main.py
"""
SIA-Meteo: Master Orchestrator.

This is the entry point for the entire project. It manages the lifecycle of the
Data Science pipeline, from raw data ingestion to model deployment readiness.

Features:
- Smart Execution: Checks for existing artifacts (data, scalers, models) to avoid redundant computation.
- Pipeline Integration: Connects Data Acquisition -> Processing -> Training -> Evaluation.
- Command Line Interface (CLI): Allows forcing specific steps via flags (e.g., --force-train).

Architecture Support:
- Supports the 9/5-parameter architecture (Temp, Hum, Pres, Wind, Rain, Timestamps).
- Ensures the Scaler is ready for the ESP32 Live Mode.
"""

import sys
import os
import argparse
import traceback
from src import config

from src.data_acquisition.data_loader import download_data
from src.data_acquisition.synthetic_generator import generate_synthetic_data
from src.processing.split_data import split_and_normalize_dataset
from src.neural_network.train import train_pipeline
from src.neural_network.evaluate import evaluate_model

def check_artifact(path: str, description: str) -> bool:
    """Helper to check ia a file exists and log the status."""
    if os.path.exists(path):
        print(f"Found {description}: {path}")
        return True
    else:
        print(f"{description} not found.")
        return False

def run_orchestrator(args):
    print("\n" + "=" * 60)
    print("   SIA-METEO: INTELLIGENT PIPELINE ORCHESTRATOR    ")
    print("=" * 60 + "\n")

    # Data acquisition (raw)
    print(">>> Phase 1: Data acquisition")
    if args.force_data or not check_artifact(config.RAW_DATA_PATH, "Raw Data"):
        print("Dowloading historical data from Open-Meteo...")
        download_data()
    else:
        print("Raw data already exists. Use --force-data to overwrite.")
    print("-" * 30)

    # Synthetic generation (Hybrid)
    print(">>> Phase 2: Synthetic data augmentation")
    if args.force_data or not check_artifact(config.HYBRID_DATA_PATH, "Hybrid Dataset"):
        print("Generating Black Swan events (storms, heatwaves, frost)...")
        generate_synthetic_data()
    else:
        print("Hybrid dataset already exists.")
    print("-" * 30)

    # Preprocessing & Scaling
    # I check for the scaler because it is critical for the ESP32 Live Mode
    print(">>> Phase 3: Preprocessing and normalization")
    scaler_exists = check_artifact(config.SCALER_PATH, "MinMax Scaler")
    train_exists = check_artifact(os.path.join(config.DATA_DIR, 'train', 'train.csv'), "Training Set")

    if args.force_data or not (scaler_exists and train_exists):
        print("Splitting data and fitting the scaler...")
        split_and_normalize_dataset()
    else:
        print("Data is already processed and normalized.")
    print("-" * 30)

    # Neural Network training
    print(">>> Phase 4: Model training (LSTM)")
    model_path = os.path.join(config.BASE_DIR, 'models', 'trained_model.keras')

    if args.force_train or not check_artifact(model_path, "Trained Model"):
        print(f"Training the LSTM model ({config.EPOCHS} epochs)...")
        train_pipeline()
    else:
        print("Trained model found. Use --force-train to retrain.")
    print("-" * 30)

    # Evaluating & Reporting
    print(">>> Phase 5: Evaluation and metrics")
    metrics_path = os.path.join(config.BASE_DIR, 'results', 'test_metrics.json')

    if args.skip_eval:
        print("Evaluation skipped by user.")
    else:
        print("Running evaluation on test set (2024)...")
        evaluate_model()
    print("-" * 30)

    print("\n" + "=" * 60)
    print("   ✅ PIPELINE COMPLETE. SYSTEM READY FOR LIVE MODE.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    # Command line interface definition
    parser = argparse.ArgumentParser(description="SIA-Meteo Pipeline Controller")

    parser.add_argument('--force-data', action='store_true',
                        help="Force re-download and re-generation of all datasets.")
    parser.add_argument('--force-train', action='store_true',
                        help="Force re-training of the Neural Network model.")
    parser.add_argument('--skip-eval', action='store_true',
                        help="Skip the evaluation phase.")

    args = parser.parse_args()

    try:
        run_orchestrator(args)
    except KeyboardInterrupt:
        print(f"\n\n[STOP] Process interupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)
