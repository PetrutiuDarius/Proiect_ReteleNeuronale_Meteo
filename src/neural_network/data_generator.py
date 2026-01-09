# src/neural_network/data_generator.py
import numpy as np
import pandas as pd
from typing import Tuple, List

class TimeSeriesGenerator:
    """
    Utlity class to transform flat tabular data into 3D sequences required by LSTM models.
    Structure: (Samples, TimeSteps, Features)
    """

    def __init__(self, input_width: int, label_width: int, feature_cols: List[str], target_col: str):
        """
        :param input_width: How many past steps (hours) to look back (e.g., 24).
        :param label_width: How many steps into the future to predict (offset).
        :param feature_cols: List of column names used as input features.
        :param target_col: Name of the column to predict
        """
        self.input_width = input_width
        self.label_width = label_width
        self.feature_cols = feature_cols
        self.target_col = target_col

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slices the DataFrame into input sequences (X) and target values (y).

        Logic:
        If we are at time 't', X is data from [t - input_width] to [t]
        y is the value a [t + label_width].
        """
        # Convert dataframe to numpy array for performance
        data_array = df[self.feature_cols].values
        target_array = df[self.target_col].values

        X, y = [], []

        # Iterate through the data, ensuring we have enough bounds for both history and future
        # Start from 'input_width' because we need prior history
        # Stop before 'len - label_width' because we need a future target
        for i in range(self.input_width, len(df) - self.label_width):

            # Extract the window of past data
            # Shape: (input_width, number_of_features)
            window = data_array[i - self.input_width : i]

            # Extract the target value in the future
            # Shape: (1,) - predicting a single scalar value
            target = target_array[i + self.label_width]

            X.append(window)
            y.append(target)

        # Convert lists to numpy arrays which Keras expects
        return np.array(X), np.array(y)

# Quick sanity check logic if run directly
if __name__ == "__main__":
    print("[TEST] Returning Data Generator test...")
    # Mock dataframe
    mock_data = pd.DataFrame(np.random.rand(100, 4), columns=['temperature', 'humidity', 'pressure', 'wind_speed'])
    gen = TimeSeriesGenerator(input_width=24, label_width=6, feature_cols=mock_data.columns, target_col='temperature')
    X_test, y_test = gen.create_sequences(mock_data)
    print(f"Input shape (Samples, TimeSteps, Feature): {X_test.shape}")
    print(f"Target shape: {y_test.shape}")
