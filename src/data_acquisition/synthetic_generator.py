# src/data_acquisition/synthetic_generator.py
import pandas as pd
import numpy as np
import os
import random
from src import config
from src.data_acquisition.data_loader import load_raw_data


def generate_heatwave(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates extreme heatwaves based on historical summer patterns.
    Augments temperature to exceed historical maximums (Config: >42°C).
    """
    print("[GEN] Generating Heatwaves...")
    # Select summer months (June, July, August) from historical data as base
    summer_data = base_df[base_df.index.month.isin([6, 7, 8])].sample(n=2000, replace=True).copy()

    # Apply synthetic transformation
    # We boost temperature and lower humidity (hot & dry)
    summer_data['temperature'] = np.random.uniform(
        config.EXTREME_SCENARIOS['heatwave']['temp_min'],
        config.EXTREME_SCENARIOS['heatwave']['temp_max'],
        size=len(summer_data)
    )
    summer_data['humidity'] = summer_data['humidity'] * 0.6  # Lower humidity
    summer_data['is_simulated'] = 1

    return summer_data


def generate_storm(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates severe storms with high wind speeds and pressure drops.
    """
    print("[GEN] Generating Severe Storms...")
    # Select data with already moderate wind as base
    storm_base = base_df.sample(n=2000, replace=True).copy()

    # Synthesize extreme wind and pressure drop
    storm_base['wind_speed'] = np.random.uniform(
        config.EXTREME_SCENARIOS['storm']['wind_speed_min'],
        config.EXTREME_SCENARIOS['storm']['wind_speed_max'],
        size=len(storm_base)
    )
    # Drop pressure significantly
    storm_base['pressure'] = storm_base['pressure'] - config.EXTREME_SCENARIOS['storm']['pressure_drop']
    storm_base['is_simulated'] = 1

    return storm_base


def generate_late_frost(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates freezing temperatures in spring (April/May).
    Crucial for agriculture protection use-case.
    """
    print("[GEN] Generating Late Frost events...")
    # Filter spring months
    spring_months = config.EXTREME_SCENARIOS['late_frost']['months']
    spring_base = base_df[base_df.index.month.isin(spring_months)].sample(n=1000, replace=True).copy()

    # Force negative temperatures
    spring_base['temperature'] = np.random.uniform(
        config.EXTREME_SCENARIOS['late_frost']['temp_min'],
        config.EXTREME_SCENARIOS['late_frost']['temp_max'],
        size=len(spring_base)
    )
    spring_base['is_simulated'] = 1

    return spring_base


def generate_synthetic_data():
    """Main function to coordinate generation."""
    print(f"[INFO] Starting Synthetic Data Generation (Target: ~{config.SYNTHETIC_SAMPLES_TARGET} samples)...")

    real_df = load_raw_data()

    # 1. Generate Scenarios
    df_heat = generate_heatwave(real_df)
    df_storm = generate_storm(real_df)
    df_frost = generate_late_frost(real_df)

    # 2. Fill the rest with slightly modified noise data to reach target size
    # This ensures the model sees enough "simulated" data to learn the label
    remaining_count = config.SYNTHETIC_SAMPLES_TARGET - (len(df_heat) + len(df_storm) + len(df_frost))
    if remaining_count > 0:
        print(f"[GEN] Generating {remaining_count} generic synthetic samples (noise injection)...")
        noise_df = real_df.sample(n=remaining_count, replace=True).copy()
        # Add small Gaussian noise to make it distinct
        noise = np.random.normal(0, 0.5, size=len(noise_df))
        noise_df['temperature'] += noise
        noise_df['is_simulated'] = 1
    else:
        noise_df = pd.DataFrame()

    # 3. Combine All
    synthetic_df = pd.concat([df_heat, df_storm, df_frost, noise_df])

    # Save Synthetic Only
    os.makedirs(os.path.dirname(config.GENERATED_DATA_PATH), exist_ok=True)
    synthetic_df.to_csv(config.GENERATED_DATA_PATH)
    print(f"[SUCCESS] Synthetic dataset saved: {config.GENERATED_DATA_PATH}")

    # 4. Create Hybrid Dataset (Real + Synthetic)
    print("[INFO] Merging Real and Synthetic datasets...")
    hybrid_df = pd.concat([real_df, synthetic_df])

    # Sort by timestamp isn't strictly possible because synthetic data is random in time,
    # but we append it. For LSTM we might need to be careful, but for now we append.
    # A better strategy for LSTM is to inject these as episodes.
    # For this stage, we simply concatenate.

    hybrid_df.to_csv(config.HYBRID_DATA_PATH)
    print(f"[SUCCESS] Hybrid dataset ready: {config.HYBRID_DATA_PATH}")
    print(f"Total Samples: {len(hybrid_df)} (Real: {len(real_df)} | Synthetic: {len(synthetic_df)})")


if __name__ == "__main__":
    generate_synthetic_data()