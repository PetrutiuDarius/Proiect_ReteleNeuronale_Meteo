# src/generate_docs.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src import config


def generate_documentation_assets():
    print("[DOCS] Generare statistici și grafice pentru README...")

    # 1. Încărcăm datele hibride
    if not os.path.exists(config.HYBRID_DATA_PATH):
        print("Eroare: Nu găsesc hybrid_dataset.csv. Rulează main.py întâi.")
        return

    df = pd.read_csv(config.HYBRID_DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Separam Real vs Simulat
    df_real = df[df['is_simulated'] == 0]
    df_sim = df[df['is_simulated'] == 1]

    # --- A. TABEL STATISTICI (Printat în consolă pentru Copy-Paste) ---
    print("\n" + "=" * 50)
    print("   TABEL STATISTICI (Copiaza asta in README)")
    print("=" * 50)
    print("| Anul | Tip Date | Temp Max (°C) | Vânt Max (m/s) | Presiune Min (hPa) |")
    print("|------|----------|---------------|----------------|--------------------|")

    # Statistici pe ani reali
    for year in sorted(df_real['timestamp'].dt.year.unique()):
        d_year = df_real[df_real['timestamp'].dt.year == year]
        print(
            f"| {year} | Real | {d_year['temperature'].max():.1f} | {d_year['wind_speed'].max():.1f} | {d_year['pressure'].min():.1f} |")

    # Statistici date simulate
    print(
        f"| Simulat | Sintetic | **{df_sim['temperature'].max():.1f}** | **{df_sim['wind_speed'].max():.1f}** | **{df_sim['pressure'].min():.1f}** |")
    print("=" * 50 + "\n")

    # --- B. GRAFIC COMPARATIV (Real vs Simulat) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_real['temperature'], label='Date Reale (Istoric)', fill=True, color='blue', alpha=0.3)
    sns.kdeplot(df_sim['temperature'], label='Date Simulate (Extreme)', fill=True, color='red', alpha=0.3)
    plt.title('Distribuția Temperaturii: Istoric vs. Scenarii Extreme')
    plt.xlabel('Temperatură (°C)')
    plt.ylabel('Densitate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(config.BASE_DIR, 'docs', 'distribution_comparison.png')
    plt.savefig(output_path)
    print(f"[SUCCESS] Grafic salvat în: {output_path}")


if __name__ == "__main__":
    generate_documentation_assets()