# src/data_loader.py
import pandas as pd
import requests
import os
from sklearn.preprocessing import MinMaxScaler

# URL pentru datele meteo Bucuresti (2020-2024)
DATA_URL = "https://archive-api.open-meteo.com/v1/archive?latitude=44.4323&longitude=26.1063&start_date=2020-01-01&end_date=2024-01-01&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&timezone=Europe%2FBucharest&format=csv&wind_speed_unit=ms"

# Identificam caile relative corecte
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'bucharest_weather_raw.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'bucharest_normalized.csv')


def download_data():
    """Descarca datele de pe internet daca nu exista deja."""
    if os.path.exists(RAW_DATA_PATH):
        print(f"Fisierul brut exista deja: {RAW_DATA_PATH}")
        return

    print("Se descarca datele meteo pentru Bucuresti...")
    response = requests.get(DATA_URL)

    # Ne asiguram ca folderul exista
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

    with open(RAW_DATA_PATH, 'wb') as f:
        f.write(response.content)
    print(f"Date brute salvate cu succes.")


def process_data():
    """Curata si normalizeaza datele."""
    print("Se proceseaza datele...")

    # 1. Citire (sarim peste primele 2 randuri de metadata de la OpenMeteo)
    try:
        df = pd.read_csv(RAW_DATA_PATH, skiprows=2)
    except FileNotFoundError:
        print("Eroare: Nu gasesc fisierul brut. Ruleaza download_data() intai.")
        return

    # 2. Redenumire coloane (sa fie usor de lucrat)
    rename_map = {
        'time': 'timestamp',
        'temperature_2m (°C)': 'temperature',
        'relative_humidity_2m (%)': 'humidity',
        'surface_pressure (hPa)': 'pressure',
        'wind_speed_10m (m/s)': 'wind_speed'
    }

    df.rename(columns=rename_map, inplace=True)

    # Verificam daca redenumirea a functionat pentru toate coloanele
    required_cols = ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed']

    # Verificare de siguranta
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"EROARE CRITICA: Nu gasesc coloanele: {missing_cols}")
        print(f"Coloanele actuale din fisier sunt: {df.columns.tolist()}")
        print("SFAT: Sterge fisierul din data/raw/ si ruleaza din nou main.py pentru a redescărca datele corecte.")
        return

    # 3. Pastram doar ce ne trebuie si setam Timpul ca Index
    df = df[required_cols]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 4. Gestionare valori lipsa (Interpolare - vremea nu sare brusc)
    df.interpolate(method='time', inplace=True)

    # 5. Normalizare (aducem totul intre 0 si 1 pentru reteaua neuronala)
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # 6. Salvare
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_normalized.to_csv(PROCESSED_DATA_PATH)

    print(f"Fisierul procesat este aici: {PROCESSED_DATA_PATH}")
    print(f"Dimensiune dataset: {df_normalized.shape[0]} ore (randuri)")
    print("\nPrimele 5 randuri normalizate:")
    print(df_normalized.head())