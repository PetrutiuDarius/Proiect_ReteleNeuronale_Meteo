# src/generate_eda.py
"""
Exploratory Data Analysis (EDA) Module.

This script performs statistical analysis and visualization on the RAW historical dataset.
It generates the necessary assets (histograms, boxplots, correlation matrices) for the
project documentation (README Etapa 3).

Usage:
    Run this script to generate PNG assets in the 'docs/' folder and print statistics
    to the console for inclusion in reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src import config


def load_and_clean_data() -> pd.DataFrame:
    """
    Loads raw weather data and cleans column names for analysis.
    Mimics the logic in data_loader.py but for analysis purposes.
    """
    if not os.path.exists(config.RAW_DATA_PATH):
        print(f"[ERROR] Raw data not found at {config.RAW_DATA_PATH}")
        print("Please run 'main.py --force-data' first.")
        sys.exit(1)

    try:
        # Open-Meteo CSVs usually have 2 lines of metadata before header
        df = pd.read_csv(config.RAW_DATA_PATH, skiprows=2)

        # Rename raw columns to clean, professional names
        # We use a flexible map in case specific units differ slightly
        rename_map = {
            'time': 'timestamp',
            'temperature_2m (°C)': 'temperature',
            'relative_humidity_2m (%)': 'humidity',
            'surface_pressure (hPa)': 'pressure',
            'wind_speed_10m (m/s)': 'wind_speed',
            'precipitation (mm)': 'precipitation'
        }

        # Rename available columns
        df.rename(columns=rename_map, inplace=True)

        # Handle Timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        print(f"[CRITICAL] Failed to load raw data: {e}")
        sys.exit(1)


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """Generates and saves distribution and correlation plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Filter Numeric Columns Only
    # This prevents the "could not convert string to float" error
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()

    if not numeric_cols:
        print("[ERROR] No numeric columns found for analysis.")
        return

    # 2. Distributions (Histograms)
    # Adjust grid size based on number of columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(df[col], kde=True, bins=30, color='#45B7D1', edgecolor='black')
        plt.title(f'Distribution: {col.capitalize()}')
        plt.xlabel(col)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'eda_distributions.png')
    plt.savefig(hist_path)
    plt.close()

    # 3. Outliers (Boxplots)
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=numeric_df, palette="Set2")
    plt.title('Outlier Detection (Boxplots)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    box_path = os.path.join(output_dir, 'eda_outliers.png')
    plt.savefig(box_path)
    plt.close()

    # 4. Correlation Matrix
    plt.figure(figsize=(8, 6))
    # Calculate correlation only on numeric dataframe
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix (Pearson)')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, 'eda_correlation.png')
    plt.savefig(corr_path)
    plt.close()

    print(f"[SUCCESS] Visualizations saved to: {output_dir}")


def print_statistics(df: pd.DataFrame):
    """Calculates and prints formatted statistics for the README."""
    print("\n" + "=" * 80)
    print("   STATISTICAL SUMMARY (Copy to README)")
    print("=" * 80)

    # Select only numeric columns for stats
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats['IQR'] = stats['75%'] - stats['25%']

    # Header
    print(f"| {'Feature':<15} | {'Mean':<8} | {'Std Dev':<8} | {'Min':<8} | {'Median':<8} | {'Max':<8} | {'IQR':<8} |")
    print("|" + "---|" * 7)

    # Rows
    for idx, row in stats.iterrows():
        print(
            f"| **{idx}** | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['50%']:.2f} | {row['max']:.2f} | {row['IQR']:.2f} |")

    print("\n[INFO] Missing Values:")
    print(df.isnull().sum())
    print("=" * 80 + "\n")


if __name__ == "__main__":
    print("[INFO] Starting EDA Process...")

    # 1. Load Data
    df = load_and_clean_data()

    # 2. Generate Plots
    docs_path = os.path.join(config.BASE_DIR, 'docs')
    generate_visualizations(df, docs_path)

    # 3. Print Stats
    print_statistics(df)