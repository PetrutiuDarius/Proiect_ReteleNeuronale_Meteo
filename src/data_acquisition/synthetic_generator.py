# src/data_acquisition/synthetic_generator.py
"""
Synthetic Data Generator Module.

This module implements statistical augmentation strategies to create "Black Swan"
events (extreme weather scenarios) that are historically rare but critical for
training robust Neural Networks.

Scenarios generated:
1. Heatwaves (high temp, low humidity)
2. Severe storms (high wind, pressure drop, heavy rain)
3. Late frost (negative temp in Spring)
"""

import pandas as pd
import numpy as np
import os
from src import config
from src.data_acquisition.data_loader import load_raw_data, add_time_features

def generate_random_timestamps(n_samples: int, months: list) -> pd.DatetimeIndex:
    """
    Generates random hourly timestamps within specific months.
    Used to place synthetic events in the correct season (e.g., Snow in Winter).
    """
    # Create a generic year timeframe (e.g., 2022)
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='h')
    # Filter by desired months
    filtered = dates[dates.month.isin(months)]
    # Sample randomly
    timestamps = pd.DatetimeIndex(np.random.choice(filtered, size=n_samples))
    timestamps.name = 'timestamp'
    return timestamps

def generate_heatwave(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates extreme heatwaves (>40°C).
    Physics: Temperature rises, Humidity drops, Zero precipitation.
    """
    print("Generating heatwaves...")

    # Summer months: June, July, August
    n = 2000
    timestamps = generate_random_timestamps(n, [6, 7, 8])

    # Creates base dataframe
    summer_data = pd.DataFrame(index=timestamps)

    # Physics
    summer_data['temperature'] = np.random.uniform(40.0, 44.0, size=n)
    summer_data['humidity'] = np.random.uniform(20.0, 40.0, size=n)  # Dry
    summer_data['pressure'] = np.random.normal(1010, 5, size=n)
    summer_data['wind_speed'] = np.random.uniform(0, 10, size=n)
    summer_data['precipitation'] = 0.0

    # Add Time Features
    summer_data = add_time_features(summer_data)

    summer_data['is_simulated'] = 1
    return summer_data[config.FEATURE_COLS + ['is_simulated']]

def generate_storm(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates severe storms.
    Physics: High wind, Sharp pressure drop, Heavy precipitation, High humidity.
    """
    print("Generating severe storms...")

    # Storms can happen anytime, mostly Spring/Autumn
    n = 2000
    timestamps = generate_random_timestamps(n, [3, 4, 5, 9, 10, 11])

    storm_base = pd.DataFrame(index=timestamps)

    # Extreme wind
    storm_base['temperature'] = np.random.uniform(10.0, 25.0, size=n)
    storm_base['wind_speed'] = np.random.uniform(20.0, 30.0, size=n)
    storm_base['pressure'] = np.random.uniform(970.0, 990.0, size=n)  # Low pressure
    storm_base['precipitation'] = np.random.uniform(1.0, 15.0, size=n)
    storm_base['humidity'] = np.random.uniform(80.0, 100.0, size=n)

    storm_base = add_time_features(storm_base)

    storm_base['is_simulated'] = 1
    return storm_base[config.FEATURE_COLS + ['is_simulated']]

def generate_late_frost(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates agricultural frost in spring.
    Physics: Sub-zero temperature in April/May, zero rain (clear sky frost).
    """
    print("Generating late frost events...")

    # April/May
    n = 1000
    timestamps = generate_random_timestamps(n, [4, 5])

    frost_base = pd.DataFrame(index=timestamps)

    frost_base['temperature'] = np.random.uniform(-3.0, 0.0, size=n)
    frost_base['precipitation'] = 0.0
    frost_base['humidity'] = np.random.uniform(40.0, 70.0, size=n)
    frost_base['pressure'] = np.random.uniform(1015.0, 1030.0, size=n)  # High pressure
    frost_base['wind_speed'] = np.random.uniform(0.0, 5.0, size=n)

    frost_base = add_time_features(frost_base)

    frost_base['is_simulated'] = 1
    return frost_base[config.FEATURE_COLS + ['is_simulated']]

def generate_synthetic_data():
    """
    Orchestrates the generation pipeline and merges with real data.
    Ensures the training set has enough 'bad weather' examples for the AI to learn.
    """
    print(f"Starting synthetic data pipeline. Target: ~{config.SYNTHETIC_SAMPLES_TARGET} samples.")

    real_df = load_raw_data()

    # Generate specific scenarios
    df_heat = generate_heatwave(real_df)
    df_storm = generate_storm(real_df)
    df_frost = generate_late_frost(real_df)

    # Noise injection (general data augmentation)
    # I create slightly modified versions of real data to improve model robustness
    # against sensor noise (e.g., from the ESP32)
    current_count = len(df_heat) + len(df_storm) + len(df_frost)
    remaining = config.SYNTHETIC_SAMPLES_TARGET - current_count

    if remaining > 0:
        print(f"[GEN] Augmenting with {remaining} noise samples...")
        noise_df = real_df.sample(n=remaining, replace=True).copy()

        # Add noise only to physical columns, not time columns
        noise_df['temperature'] += np.random.normal(0, 0.5, size=len(noise_df))
        noise_df['humidity'] = (noise_df['humidity'] + np.random.normal(0, 2.0, size=len(noise_df))).clip(0, 100)

        noise_df['is_simulated'] = 1

        # Ensure column order
        noise_df = noise_df[config.FEATURE_COLS + ['is_simulated']]
    else:
        noise_df = pd.DataFrame()

    # Combine all
    synthetic_df = pd.concat([df_heat, df_storm, df_frost, noise_df])

    # Ensure the index is named before merge/save
    synthetic_df.index.name = 'timestamp'

    # Save synthetic only
    os.makedirs(os.path.dirname(config.GENERATED_DATA_PATH), exist_ok=True)
    synthetic_df.to_csv(config.GENERATED_DATA_PATH)
    print(f"Synthetic dataset saved to: {config.GENERATED_DATA_PATH}")

    # Create the hybrid dataset (real + synthetic)
    print("Creating the hybrid dataset (real + synthetic)...")
    hybrid_df = pd.concat([real_df, synthetic_df])
    hybrid_df.index.name = 'timestamp'
    hybrid_df.to_csv(config.HYBRID_DATA_PATH)

    print(f"Hybrid dataset generated at: {config.HYBRID_DATA_PATH}")
    print(f"   > Total Samples: {len(hybrid_df)}")
    print(f"   > Real: {len(real_df)} | Synthetic: {len(synthetic_df)}")
    print(f"   > Columns: {len(hybrid_df.columns)} (Expected 10: 9 Features + 1 Flag)")

if __name__ == "__main__":
    generate_synthetic_data()