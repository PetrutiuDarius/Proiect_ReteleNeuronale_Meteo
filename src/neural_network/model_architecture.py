# src/neural_network/model_architecture.py
"""
Neural Network architecture definition.

Defines the topology of the LSTM model used for multi-target regression.
Designed to capture temporal dependencies via stacked LSTM layers and
prevent overfitting via Dropout.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape: tuple, learning_rate: float = 0.001, output_units: int = 5) -> Sequential:
    """
    Constructs and compiles the LSTM Neural Network.

    Architecture design:
    1. Input Layer: Defines the shape of the incoming date window.
    2. LSTM Layer 1: 128 units, captures high-level temporal abstractions.
    3. Dropout (0.3): Regularization to prevent overfitting on synthetic noise.
    4. LSTM Layer 2: 64 units, aggregates temporal features.
    5. Dropout (0.3): Further regularization.
    6. Dense (output_units): Final regression vector (Multi-Target).

    Args:
        input_shape (tuple): Shape of input data (e.g., (24, 5)).
        learning_rate (float): Optimizer step size.
        output_units (int): Number of predicted parameters (5).

    Returns:
        Sequential: Compiled Keras Model.
    """
    model = Sequential([
        # Explicit Input Layer for visualization
        Input(shape=input_shape),

        # Layer 1: Captures high-level temporal patterns
        # return_sequences=True is required to pass temporal data to the next LSTM layer
        LSTM(units=128, return_sequences=True, activation='tanh'),
        Dropout(rate=0.3), # Randomly drop 30% of connections to improve generalization

        # Layer 2: Refines patterns into specific features
        # return_sequences=False because the next layer (Dense) expects a flat vector
        LSTM(units=64, return_sequences=False, activation='tanh'),
        Dropout(rate=0.3),

        # Output Layer: Multi-Target Regression
        Dense(units=output_units)
    ])

    # Compilation
    # MSE (Mean Squared Error) is optimal for regression tasks where large errors should be penalized
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model
