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
import sys
from src import config
from src.data_acquisition.data_loader import load_raw_data

def generate_heatwave(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates extreme heatwaves (>40°C).
    Physics: Temperature rises, Humidity drops, Zero precipitation.
    """
    print("Generating heatwaves...")

    # Filter for summer months (June-August) to maintain seasonal realism
    summer_data = base_df[base_df.index.month.isin([6, 7, 8])].sample(n=2000, replace=True).copy()

    # Boost temperature
    summer_data['temperature'] = np.random.uniform(
        config.EXTREME_SCENARIOS['heatwave']['temp_min'],
        config.EXTREME_SCENARIOS['heatwave']['temp_max'],
        size=len(summer_data)
    )

    # Drop humidity (heatwaves are often dry)
    summer_data['humidity'] = summer_data['humidity'] * 0.6

    # No Rain during scorching heat
    summer_data['precipitation'] = 0.0

    summer_data['is_simulated'] = 1
    return summer_data

def generate_storm(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates severe storms.
    Physics: High wind, Sharp pressure drop, Heavy precipitation, High humidity.
    """
    print("Generating severe storms...")

    # Base sampling
    storm_base = base_df.sample(n=2000, replace=True).copy()

    # Extreme wind
    storm_base['wind_speed'] = np.random.uniform(
        config.EXTREME_SCENARIOS['storm']['wind_speed_min'],
        config.EXTREME_SCENARIOS['storm']['wind_speed_max'],
        size=len(storm_base)
    )

    # Pressure drop (Cyclogenesis signal)
    storm_base['pressure'] = storm_base['pressure'] - config.EXTREME_SCENARIOS['storm']['pressure_drop']

    # Heavy rain (1mm to 15mm per hour)
    storm_base['precipitation'] = np.random.uniform(1.0, 15.0, size=len(storm_base))

    # High humidity (Saturation)
    storm_base['humidity'] = np.random.uniform(85.0, 100.0, size=len(storm_base))

    storm_base['is_simulated'] = 1
    return storm_base

def generate_late_frost(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates agricultural frost in spring.
    Physics: Sub-zero temperature in April/May, zero rain (clear sky frost).
    """
    print("Generating late frost events...")

    spring_months = config.EXTREME_SCENARIOS['late_frost']['months']
    spring_base = base_df[base_df.index.month.isin(spring_months)].sample(n=1000, replace=True).copy()

    # Freeze temperature
    spring_base['temperature'] = np.random.uniform(
        config.EXTREME_SCENARIOS['late_frost']['temp_min'],
        config.EXTREME_SCENARIOS['late_frost']['temp_max'],
        size=len(spring_base)
    )

    # No rain (Frost usually happens on clear nights)
    spring_base['precipitation'] = 0.0

    spring_base['is_simulated'] = 1
    return spring_base

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
    remaining_count = config.SYNTHETIC_SAMPLES_TARGET - current_count

    if remaining_count > 0:
        print(f"Augmenting with {remaining_count} noise-injected samples...")
        noise_df = real_df.sample(n=remaining_count, replace=True).copy()

        # Add Gaussian noise to Temperature
        noise_df['temperature'] += np.random.normal(0, 0.5, size=len(noise_df))

        # Add slight noise to Humidity (clipping to 0-100%)
        noise_df['humidity'] += np.random.normal(0, 2.0, size=len(noise_df))
        noise_df['humidity'] = noise_df['humidity'].clip(0, 100)

        # Add slight noise to Pressure
        noise_df['pressure'] += np.random.normal(0, 1.0, size=len(noise_df))

        # Flag as simulated
        noise_df['is_simulated'] = 1
    else:
        noise_df = pd.DataFrame()

    # Combine all
    synthetic_df = pd.concat([df_heat, df_storm, df_frost, noise_df])

    # Save synthetic only
    os.makedirs(os.path.dirname(config.GENERATED_DATA_PATH), exist_ok=True)
    synthetic_df.to_csv(config.GENERATED_DATA_PATH)
    print(f"Synthetic dataset saved to: {config.GENERATED_DATA_PATH}")

    # Create the hybrid dataset (real + synthetic)
    print("Creating the hybrid dataset (real + synthetic)...")
    hybrid_df = pd.concat([real_df, synthetic_df])

    hybrid_df.to_csv(config.HYBRID_DATA_PATH)

    print(f"Hybrid dataset generated at: {config.HYBRID_DATA_PATH}")
    print(f"   > Total Samples: {len(hybrid_df)}")
    print(f"   > Real: {len(real_df)} | Synthetic: {len(synthetic_df)}")

if __name__ == "__main__":
    generate_synthetic_data()