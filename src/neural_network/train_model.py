# src/neural_network/train_model.py
"""
Model Training Pipeline.

Orchestrates the loading of data, sequence generation, model compilation,
and the training loop. Saves the trained artifacts and training history.
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Environment Setup (Clean Console) ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

from src import config
from src.neural_network.data_generator import TimeSeriesGenerator
from src.neural_network.model_architecture import build_lstm_model

def train_pipeline():
    print("==========================================")
    print("     STARTING NEURAL NETWORK TRAINING     ")
    print("==========================================")

    # Load data (already processed and split in stage 3/4)
    train_path = os.path.join(config.DATA_DIR, 'train', 'train.csv')
    val_path = os.path.join(config.DATA_DIR, 'validation', 'validation.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found at {train_path}. Run main.py first.")

    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Prepare sequences (sliding window)
    # Initiate generator class
    print(f"Initializing the data generator (features: {len(config.FEATURE_COLS)})...")
    gen = TimeSeriesGenerator(
        input_width=config.SEQ_LENGTH,
        label_width=config.PREDICT_HORIZON,
        feature_cols=config.FEATURE_COLS,
        target_cols=config.TARGET_COLS
    )

    print(f"Generating sequences (lookback: {config.SEQ_LENGTH}h -> predict: +{config.PREDICT_HORIZON}h)...")
    X_train, y_train = gen.create_sequences(df_train)
    X_val, y_val = gen.create_sequences(df_val)

    print(f"  -> Training samples: {X_train.shape}")
    print(f"  -> Validation samples: {X_val.shape}")

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2]) # (24, 5)
    output_units = len(config.TARGET_COLS) # 5

    model = build_lstm_model(
        input_shape=input_shape,
        learning_rate=config.LEARNING_RATE,
        output_units=output_units
    )

    print("\n Model architecture:")
    model.summary()

    # Callbacks
    # EarlyStopping: Stop if validation loss doesn't improve for 'patience' epochs
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True
    )

    # ModelCheckpoint: Save the best model during training
    model_save_path = os.path.join(config.BASE_DIR, 'models', 'trained_model.keras')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True
    )

    # Training loop
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1 # Shows the progress bar
    )

    # Save training history (loss curve)
    print("\nGenerating training loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation loss (MSE)')
    plt.title('Model Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (mean squared error)')
    plt.legend()
    plt.grid(True)

    loss_img_path = os.path.join(config.BASE_DIR, 'docs', 'loss_curve.png')
    plt.savefig(loss_img_path)
    print(f"Loss curve saved to {loss_img_path}")
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_pipeline()
