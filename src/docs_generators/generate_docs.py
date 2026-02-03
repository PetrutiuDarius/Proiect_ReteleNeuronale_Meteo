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
    unique_years = sorted(df_real['timestamp'].dt.year.dropna().unique())

    for year in unique_years:
        d_year = df_real[df_real['timestamp'].dt.year == year]

        # Check for column existence to ensure backward compatibility
        max_rain = d_year['precipitation'].max() if 'precipitation' in d_year.columns else 0.0

        print(f"| {int(year)} | Real      | "
              f"{d_year['temperature'].max():.1f}          | "
              f"{d_year['wind_speed'].max():.1f}           | "
              f"{d_year['pressure'].min():.1f}              | "
              f"{max_rain:.1f}          |")

    # Synthetic sata analysis (aggregated)
    df_sim = df[df['is_simulated'] == 1]

    if not df_sim.empty:
        max_rain = df_sim['precipitation'].max() if 'precipitation' in df_sim.columns else 0.0

        # Highlighting synthetic extremes using Bold Markdown
        print(f"| Sim   | **Synthetic** | "
              f"**{df_sim['temperature'].max():.1f}** | "
              f"**{df_sim['wind_speed'].max():.1f}** | "
              f"**{df_sim['pressure'].min():.1f}** | "
              f"**{max_rain:.1f}** |")

    print("=" * 80 + "\n")

def plot_temperature_distribution(df: pd.DataFrame) -> None:
    """
    Generates a KDE (Kernel Density Estimate) plot to visualize the distribution shift
    introduced by the synthetic data (handling the 'imbalanced dataset' problem).
    """
    print("Generating distribution comparison plot...")

    plt.figure(figsize=(10, 6))

    # Plot real data distribution
    sns.kdeplot(
        df[df['is_simulated'] == 0]['temperature'],
        label='Real Data (Historical)',
        fill=True, color='blue', alpha=0.3
    )

    # Plot synthetic data distribution
    if not df[df['is_simulated'] == 1].empty:
        sns.kdeplot(
            df[df['is_simulated'] == 1]['temperature'],
            label='Synthetic Data (Extremes)',
            fill=True, color='red', alpha=0.3
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

    # Generate stats tablea
    generate_statistics_table(df)

    # Generate visualizations
    plot_temperature_distribution(df)

if __name__ == "__main__":
    main()