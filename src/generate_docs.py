# src/generate_docs.py
"""
Documentation asset generator module.

This script is responsible for generating statistical insights and visualizations
from the hybrid dataset.
It produces:
1. A markdown-formatted statistical table printed to the console.
2. A distribution plot comparing Real vs. Synthetic data saved to 'docs/'.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from src import config

def load_hybrid_data() -> pd.DataFrame:
    """
    Loads the hybrid dataset and ensures the timestamp is properly typed.
    :return: pd.DataFrame: The loaded dataset with datetime index.
    """
    if not os.path.exists(config.HYBRID_DATA_PATH):
        print(f"Hybrid dataset not found at {config.HYBRID_DATA_PATH}")
        print("Please run 'main.py' fisrt to generate the data.")
        sys.exit(1)

    try:
        df = pd.read_csv(config.HYBRID_DATA_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

def generate_statistics_table(df: pd.DataFrame) -> None:
    """
    Calculates and prints a markdown-formatted table comparing annual statistics.
    It highlights the difference between historical real data and the generated synthetic extremes.
    """
    print("\n" + "=" * 80)
    print("   DATASET STATISTICS (to copy into README.md)")
    print("=" * 80)

    print("| Year | Data Type | Max Temp (°C) | Max Wind (m/s) | Min Pressure (hPa) | Max Rain (mm) |")
    print("|------|-----------|---------------|----------------|--------------------|---------------|")

    # Real data analysis (grouped by year)
    df_real = df[df['is_simulated'] == 0]

    # Iterate through unique years in the real dataset
    unique_years = sorted(df_real['timestamp'].dt.year.dropna().unique())
    for year in unique_years:
        d_year = df_real[df_real['timestamp'].dt.year == year]

        max_temp = d_year['temperature'].max()
        max_wind = d_year['wind_speed'].max()
        min_pres = d_year['pressure'].min()
        max_rain = d_year['precipitation'].max() if 'precipitation' in d_year.columns else 0.0

        print(
            f"| {int(year)} | Real      | {max_temp:.1f}          | {max_wind:.1f}           | {min_pres:.1f}              | {max_rain:.1f}          |")

    # Synthetic sata analysis (aggregated)
    df_sim = df[df['is_simulated'] == 1]

    if not df_sim.empty:
        sim_max_temp = df_sim['temperature'].max()
        sim_max_wind = df_sim['wind_speed'].max()
        sim_min_pres = df_sim['pressure'].min()
        sim_max_rain = df_sim['precipitation'].max() if 'precipitation' in df_sim.columns else 0.0

        # Using the bold Markdown (**) to highlight the extremes provided by simulation
        print(f"| Sim   | **Synthetic** | **{sim_max_temp:.1f}** | **{sim_max_wind:.1f}** | **{sim_min_pres:.1f}** | **{sim_max_rain:.1f}** |")

    print("=" * 80 + "\n")


def plot_temperature_distribution(df: pd.DataFrame) -> None:
    """
    Generates a KDE (Kernel Density Estimate) plot to visualize the distribution shift
    introduced by the synthetic data (handling the 'imbalanced dataset' problem).
    """
    print("Generating distribution comparison plot...")

    df_real = df[df['is_simulated'] == 0]
    df_sim = df[df['is_simulated'] == 1]

    plt.figure(figsize=(10, 6))

    # Plot real data distribution
    sns.kdeplot(
        df_real['temperature'],
        label='Real data (Historical)',
        fill=True,
        color='blue',
        alpha=0.3
    )

    # Plot synthetic data distribution
    if not df_sim.empty:
        sns.kdeplot(
            df_sim['temperature'],
            label='Synthetic data (Extremes)',
            fill=True,
            color='red',
            alpha=0.3
        )

    plt.title('Temperature distribution: Historical vs. Synthetic scenarios')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Ensure docs directory exists
    os.makedirs(os.path.dirname(config.BASE_DIR), exist_ok=True)

    # Save the asset
    output_path = os.path.join(config.BASE_DIR, 'docs', 'distribution_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()  # Close plot to free memory


def main():
    print(">>> Starting documentation asset generation...")

    # Load data
    df = load_hybrid_data()

    # Generate stats table
    generate_statistics_table(df)

    # Generate visualizations
    plot_temperature_distribution(df)

if __name__ == "__main__":
    main()