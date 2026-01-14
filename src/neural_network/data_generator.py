# src/neural_network/data_generator.py
"""
Time series data generator module.

This module provides a utility class to transform flat, 2D tabular data (pandas DataFrame)
into 3D sequences (Samples, TimeSteps, Features) required for LSTM training.
It handles the 'Sliding Window' logic crucial for temporal forecasting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

class TimeSeriesGenerator:
    """
    Generator class for creating sliding window sequences.

    Attributes:
        input_width (int): Historical lookback window size.
        label_width (int): Forecast horizon (offset into the future).
        feature_cols (List[str]): Input feature names.
        target_cols (List[str]): Target feature names to predict.
    """

    def __init__(self, input_width: int, label_width: int, feature_cols: List[str], target_cols: List[str]):
        self.input_width = input_width
        self.label_width = label_width
        self.feature_cols = feature_cols
        self.target_cols = target_cols

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slices the DataFrame into input sequences (X) and target vectors (y).

        Logic:
            At time 't', 'X' contains data from [t - input_width] to [t].
            'y' contains the target vector at [t + label_width].

        Args:
            df (pd.DataFrame): Normalized source dataframe.

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (Input), y (Labels) ready for Keras.
        """
        # Convert dataframe to a numpy array for performance
        data_array = df[self.feature_cols].values
        target_array = df[self.target_cols].values

        X, y = [], []

        # Iterate ensuring boundary safety for both history lookback and future forecast
        # Stop condition: len(df) - label_width ensures not indexing out of bounds in the future
        for i in range(self.input_width, len(df) - self.label_width):

            # Extract the window of the past data
            # Shape: (input_width, num_features) -> e.g., (24, 5)
            window = data_array[i - self.input_width : i]

            # Extract the target value in the future
            # Shape: (num_targets,) -> e.g., (5,) vector [Temp, Hum, Pres, Wind, Rain]
            target = target_array[i + self.label_width]

            X.append(window)
            y.append(target)

        return np.array(X), np.array(y)

if __name__ == "__main__":
    # Unit test
    print("Returning data generator test...")
    cols_in = ['T', 'H', 'P', 'W', 'R', 'D_sin', 'D_cos', 'Y_sin', 'Y_cos']
    cols_out = ['T', 'H', 'P', 'W', 'R']
    mock_data = pd.DataFrame(np.random.rand(100, 9), columns=cols)

    gen = TimeSeriesGenerator(24, 1, cols_in, cols_out)
    X, y = gen.create_sequences(mock_data)

    print(f"Input Shape: {X.shape} (Expected: (75, 24, 9))")
    print(f"Target Shape: {y.shape} (Expected: (75, 5))")
