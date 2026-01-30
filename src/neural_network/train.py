# src/neural_network/train.py
"""
Model Training Pipeline.

This module orchestrates the entire training lifecycle of the Neural Network.
It handles data loading, sequence generation, custom loss function definition,
model compilation, and the execution of the training loop with callbacks.

Key Features:
- Asymmetric Loss Function: Custom logic to handle zero-inflated precipitation data.
- Artifact Management: Autosaves best models and training history for analysis.
- Robust Error Handling: Ensures data prerequisites are met before execution.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Environment Setup (Clean Console) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

from src import config
from src.neural_network.data_generator import TimeSeriesGenerator
from src.neural_network.model import build_lstm_model


# -------------------------------------------------------------------------
# CUSTOM LOSS FUNCTION STRATEGY
# -------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
def asymmetric_precipitation_loss(y_true, y_pred):
    """
    Custom Loss Function: 'The Disciplinarian'.

    Standard MSE is symmetric (punishes under-prediction and over-prediction equally).
    For precipitation (sparse data), over-prediction (hallucinating rain) creates
    noise that ruins the R2 score.

    This function applies a heavy penalty factor when the model predicts HIGHER
    than the actual value for the precipitation channel, effectively suppressing
    false positives.

    Args:
        y_true: Tensor of true values.
        y_pred: Tensor of predicted values.

    Returns:
        Weighted Mean Squared Error tensor.
    """
    # 1. Calculate standard Squared Error
    squared_error = tf.square(y_true - y_pred)

    # 2. Identify Over-estimation (Model says Rain > Reality)
    # Returns 1.0 where prediction > truth, else 0.0
    overestimation_mask = tf.cast(tf.greater(y_pred, y_true), tf.float32)

    # 3. Target specific column (Precipitation is at Index 4)
    # We create a mask [0, 0, 0, 0, 1] broadcasted across the batch
    # config.TARGET_COLS has 5 elements; index 4 is rain.
    rain_col_idx = 4
    feature_count = 5
    rain_column_mask = tf.one_hot(indices=[rain_col_idx] * tf.shape(y_true)[0], depth=feature_count)

    # Reshape might be necessary to ensure broadcasting matches (Batch, Features)
    rain_column_mask = tf.reshape(rain_column_mask, tf.shape(y_true))

    # 4. Calculate Penalty Factor
    # If it is the Rain Column AND it is Over-estimated -> Apply Penalty (e.g., 20x)
    # Otherwise -> Penalty is 1.0 (Standard MSE behavior)
    penalty_magnitude = 20.0
    penalty_factor = 1.0 + (overestimation_mask * rain_column_mask * penalty_magnitude)

    # 5. Compute final loss
    return tf.reduce_mean(squared_error * penalty_factor)


# -------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# -------------------------------------------------------------------------
def train_pipeline():
    print("==========================================")
    print("     STARTING NEURAL NETWORK TRAINING     ")
    print("     (Strategy: Asymmetric Custom Loss)   ")
    print("==========================================")

    # 1. Load Data Artifacts
    # Data must be processed by 'split_data.py' before reaching this stage.
    train_path = os.path.join(config.DATA_DIR, 'train', 'train.csv')
    val_path = os.path.join(config.DATA_DIR, 'validation', 'validation.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}. Run main.py --force-data first.")

    print(f"Loading datasets from {config.DATA_DIR}...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # 2. Data Generators (Sliding Window)
    # Transforms 2D tabular data into 3D sequences (Samples, TimeSteps, Features)
    print(f"Initializing data generators (Lookback: {config.SEQ_LENGTH}h)...")

    gen = TimeSeriesGenerator(
        input_width=config.SEQ_LENGTH,
        label_width=config.PREDICT_HORIZON,
        feature_cols=config.FEATURE_COLS,
        target_cols=config.TARGET_COLS
    )

    X_train, y_train = gen.create_sequences(df_train)
    X_val, y_val = gen.create_sequences(df_val)

    print(f"  -> Training tensor shape:   {X_train.shape}")
    print(f"  -> Validation tensor shape: {X_val.shape}")

    # 3. Model Initialization
    input_shape = (X_train.shape[1], X_train.shape[2])  # e.g., (24, 9)
    output_units = len(config.TARGET_COLS)  # e.g., 5

    model = build_lstm_model(
        input_shape=input_shape,
        learning_rate=config.LEARNING_RATE,
        output_units=output_units
    )

    # 4. Compilation with Custom Loss
    # We override the default loss to use our Asymmetric logic
    print("\n[CONFIG] Compiling model with 'asymmetric_precipitation_loss'...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=asymmetric_precipitation_loss,
        metrics=['mae']  # We track MAE for human-readable error monitoring
    )

    print("\nModel Architecture:")
    model.summary()

    # 5. Define Callbacks
    # EarlyStopping: Prevents overfitting by stopping when validation loss stagnates
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    # ModelCheckpoint: Always saves the best version of the model
    model_save_path = os.path.join(config.BASE_DIR, 'models', 'trained_model.keras')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )

    # 6. Execution Loop
    print(f"\nStarting training loop for {config.EPOCHS} epochs (Batch size: {config.BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # 7. Post-Training Analysis & Artifacts
    print("\n[POST-PROCESS] Saving training history and artifacts...")

    # Ensure results directory
    results_dir = os.path.join(config.BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save History to CSV
    history_df = pd.DataFrame(history.history)
    history_df.index.name = 'epoch'
    history_csv_path = os.path.join(results_dir, 'training_history.csv')
    history_df.to_csv(history_csv_path)
    print(f"   -> History saved to: {history_csv_path}")

    # Generate Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss (Asymmetric)')
    plt.plot(history.history['val_loss'], label='Val Loss (Asymmetric)')
    plt.title('Training Convergence with Custom Loss Strategy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Magnitude')
    plt.legend()
    plt.grid(True, alpha=0.3)

    loss_img_path = os.path.join(config.BASE_DIR, 'docs', 'loss_curve.png')
    os.makedirs(os.path.dirname(loss_img_path), exist_ok=True)
    plt.savefig(loss_img_path)
    print(f"   -> Loss curve saved to: {loss_img_path}")
    print(f"   -> Best model saved to: {model_save_path}")


if __name__ == "__main__":
    train_pipeline()