# src/neural_network/model_architecture.py
import tensorflow as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape: tuple, learning_rate: float = 0.001) -> Sequential:
    """
    Constructs and compiles the LSTM Neural Network.

    Architecture design:
    1. Input Layer: Defines the shape of the incoming date window.
    2. LSTM Layer 1: 64 units, returns sequences to stack another LSTM layer.
    3. Dropout: Regularization to prevent overfitting on synthetic noise.
    4. LSTM Layer 2: 32 units, aggregates temporal features.
    5. Dense output: Single neuron for regression (predicting temperature)

    :param input_shape: Tuple (TimeSteps, Features) -> e.g., (24, 4)
    :param learning_rate: Optimizer step size.
    :return: Compiled Keras Model.
    """
    model = Sequential([
        # Explicit Input Layer for visualization
        Input(shape=input_shape),

        # Layer 1: Captures high-level temporal patterns
        # return_sequences=True is mandatory when stacking LSTMs
        LSTM(units=64, return_sequences=True, activation='tanh'),
        Dropout(rate=0.2), # Randomly drop 20% of connections to improve generalization

        # Layer 2: Refines patterns into specific features
        # return_sequences=False because the next layer is Dense (needs flat input)
        LSTM(units=32, return_sequences=False, activation='tanh'),
        Dropout(rate=0.2),

        # Output Layer: Regression (Linear activation is default for Dense)
        Dense(units=1)
    ])

    # Compilation
    # MSE (Mean Squared Error) is the standard loss for regression problems
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
