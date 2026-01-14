# src/app/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import requests
import sys
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import config

# --- SETUP ---
st.set_page_config(page_title="SIA-Meteo AI Dashboard", page_icon="⛈️", layout="wide")

# --- CUSTOM CSS (Pentru aspect profesional) ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .alert-critical { background-color: #ffcccc; color: #990000; border: 1px solid #990000; }
    .alert-warning { background-color: #fff3cd; color: #856404; border: 1px solid #856404; }
    </style>
""", unsafe_allow_html=True)


# --- UTILS ---
@st.cache_resource
def load_ai_core():
    """Loads Model and Scaler once."""
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        st.error("🚨 Modelul sau Scalerul lipsesc! Rulează 'main.py' mai întâi.")
        st.stop()
    model = load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    return model, scaler


def calculate_time_features(timestamp):
    """Computes sin/cos features for a single timestamp."""
    day = 24 * 60 * 60
    year = 365.2425 * day
    ts_s = timestamp.timestamp()

    return [
        np.sin(ts_s * (2 * np.pi / day)),
        np.cos(ts_s * (2 * np.pi / day)),
        np.sin(ts_s * (2 * np.pi / year)),
        np.cos(ts_s * (2 * np.pi / year))
    ]


def get_live_data(lat, lon):
    """Fetches last 24h of data from Open-Meteo to build history buffer."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&past_days=1&forecast_days=1&hourly=temperature_2m,relative_humidity_2m,"
           f"surface_pressure,wind_speed_10m,precipitation&timezone=auto")

    try:
        r = requests.get(url, timeout=5).json()
    except Exception as e:
        st.error(f"Eroare conexiune API: {e}")
        st.stop()

    # Process into DataFrame
    data = {
        'timestamp': pd.to_datetime(r['hourly']['time']),
        'temperature': r['hourly']['temperature_2m'],
        'humidity': r['hourly']['relative_humidity_2m'],
        'pressure': r['hourly']['surface_pressure'],
        'wind_speed': r['hourly']['wind_speed_10m'],
        'precipitation': r['hourly']['precipitation']
    }
    df = pd.DataFrame(data)

    # Filter: Get exactly last 24h up to current hour
    current_time = pd.Timestamp.now().floor('h')
    df = df[df['timestamp'] <= current_time].tail(24)

    return df.reset_index(drop=True)


# --- PREDICTION ENGINE (Autoregressive) ---
def forecast_next_24h(model, scaler, initial_sequence_24h):
    """
    Predicts hour-by-hour for the next 24h.
    Uses the model's output as input for the next step.
    """
    predictions = []
    current_seq = initial_sequence_24h.copy()  # Shape (24, 9)

    # Last known timestamp
    last_timestamp = pd.Timestamp.now().floor('h')

    for i in range(24):
        # 1. Scale input (current_seq is already scaled)
        input_tensor = np.array([current_seq])  # (1, 24, 9)

        # 2. Predict next step (Physical only, 5 cols)
        pred_scaled_5 = model.predict(input_tensor, verbose=0)[0]

        # 3. Denormalize
        dummy = np.zeros((1, 9))
        dummy[:, :5] = pred_scaled_5
        pred_real_5 = scaler.inverse_transform(dummy)[0, :5]

        # Physics Constraints & Logic
        temp = pred_real_5[0]
        hum = min(max(pred_real_5[1], 0), 100)
        pres = pred_real_5[2]
        wind = max(pred_real_5[3], 0)
        rain = max(pred_real_5[4], 0)
        if rain < 0.1: rain = 0.0  # Noise gate

        # Snow Logic
        is_snow = (rain > 0) and (temp <= config.SNOW_TEMP_THRESHOLD)
        precip_type = "❄️ Ninsoare" if is_snow else ("🌧️ Ploaie" if rain > 0 else "☁️ Noros/Senin")

        next_time = last_timestamp + timedelta(hours=i + 1)
        predictions.append({
            'Ora': next_time.strftime('%H:%M'),
            'Data': next_time.strftime('%Y-%m-%d'),
            'Temp (°C)': round(temp, 1),
            'Umiditate (%)': round(hum, 1),
            'Presiune (hPa)': round(pres, 1),
            'Vânt (m/s)': round(wind, 1),
            'Precipitații (mm)': round(rain, 2),
            'Condiție': precip_type
        })

        # 4. Update Sequence for next step (Autoregression)
        new_time_feats = calculate_time_features(next_time)
        row_real_9 = np.concatenate([pred_real_5, new_time_feats])
        row_scaled_9 = scaler.transform([row_real_9])[0]

        current_seq = np.vstack([current_seq[1:], row_scaled_9])

    return pd.DataFrame(predictions)


def analyze_alerts(df):
    """Scans the 24h forecast for extreme events."""
    alerts = []

    max_temp = df['Temp (°C)'].max()
    min_pres = df['Presiune (hPa)'].min()
    max_wind = df['Vânt (m/s)'].max()
    max_rain = df['Precipitații (mm)'].max()

    # Logic Alerts
    if max_temp > 38.0:
        alerts.append(("🔥 CANICULĂ EXTREMĂ", f"Temperatura va atinge {max_temp}°C."))
    elif max_temp > 35.0:
        alerts.append(("🟠 AVERTIZARE CĂLDURĂ", f"Temperatura ridicată: {max_temp}°C."))

    if max_wind > 20.0:
        alerts.append(("🌪️ FURTUNĂ VIOLENTĂ", f"Rafale de vânt de {max_wind} m/s."))
    elif max_wind > 12.0:
        alerts.append(("💨 VÂNT PUTERNIC", f"Vânt susținut de {max_wind} m/s."))

    if min_pres < 990:
        alerts.append(("📉 PRESIUNE CRITICĂ", "Posibilă ciclogeneză (furtună majoră)."))

    if max_rain > 10.0:
        alerts.append(("⛈️ PLOI TORENȚIALE", f"Acumulări de {max_rain} mm/h."))
    elif max_rain > 0.0 and df['Temp (°C)'].min() <= 0.5:
        alerts.append(("❄️ RISC DE ÎNGHEȚ/ZĂPADĂ", "Condiții de polei sau ninsoare."))

    return alerts


# --- COMPONENT: RESULT DISPLAY ---
def display_results(current_conditions, forecast_df, city_name):
    """Reusable component to show metrics, alerts, charts, and table."""

    st.divider()

    # 1. CURRENT CONDITIONS ROW
    st.subheader(f"📍 Condiții Actuale: {city_name}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temperatură", f"{current_conditions['temperature'].iloc[-1]:.1f} °C")
    c2.metric("Umiditate", f"{current_conditions['humidity'].iloc[-1]:.0f} %")
    c3.metric("Presiune", f"{current_conditions['pressure'].iloc[-1]:.0f} hPa")
    c4.metric("Vânt", f"{current_conditions['wind_speed'].iloc[-1]:.1f} m/s")
    rain_val = current_conditions['precipitation'].iloc[-1]
    c5.metric("Precipitații", f"{rain_val:.1f} mm")

    st.divider()

    # 2. ALERTS SECTION
    alerts = analyze_alerts(forecast_df)
    if alerts:
        st.subheader("⚠️ Situații Extreme Detectate (Următoarele 24h)")
        for title, msg in alerts:
            if "CRITICĂ" in title or "VIOLENTĂ" in title or "CANICULĂ" in title:
                st.markdown(f"<div class='alert-box alert-critical'>{title}: {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-box alert-warning'>{title}: {msg}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Prognoza arată condiții stabile pentru următoarele 24h.")

    st.divider()

    # 3. CHARTS & TABLE TAB
    st.subheader(f"📅 Prognoză Detaliată (24 Ore)")

    tab_chart, tab_table = st.tabs(["📉 Grafice Evoluție", "📄 Tabel Date"])

    with tab_chart:
        # Combo Chart: Temp (Line) + Rain (Bar)
        fig = go.Figure()

        # Temperature Line
        fig.add_trace(go.Scatter(
            x=forecast_df['Ora'], y=forecast_df['Temp (°C)'],
            name='Temperatură', line=dict(color='#ff7f0e', width=3), mode='lines+markers'
        ))

        # Rain Bars
        fig.add_trace(go.Bar(
            x=forecast_df['Ora'], y=forecast_df['Precipitații (mm)'],
            name='Precipitații', yaxis='y2', marker_color='#1f77b4', opacity=0.4
        ))

        fig.update_layout(
            title='Evoluție Temperatură vs. Precipitații',
            xaxis_title='Ora',
            yaxis=dict(title='Temperatură (°C)', side='left'),
            yaxis2=dict(title='Precipitații (mm)', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Wind & Pressure Chart
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast_df['Ora'], y=forecast_df['Presiune (hPa)'], name='Presiune',
                                  line=dict(color='purple')))
        fig2.add_trace(go.Scatter(x=forecast_df['Ora'], y=forecast_df['Vânt (m/s)'], name='Vânt',
                                  line=dict(color='green', dash='dot'), yaxis='y2'))

        fig2.update_layout(
            title='Corelație Presiune - Vânt',
            yaxis=dict(title='Presiune (hPa)'),
            yaxis2=dict(title='Vânt (m/s)', overlaying='y', side='right'),
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab_table:
        # Styled Table
        st.dataframe(
            forecast_df,
            column_config={
                "Temp (°C)": st.column_config.NumberColumn(format="%.1f °C"),
                "Umiditate (%)": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
                "Vânt (m/s)": st.column_config.NumberColumn(format="%.1f m/s"),
                "Precipitații (mm)": st.column_config.NumberColumn(format="%.2f mm"),
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )


# --- PAGES ---
def page_romania_live(model, scaler):
    st.header("🇷🇴 Monitorizare Live România")

    cities = {
        "București": (44.43, 26.10),
        "Cluj-Napoca": (46.77, 23.60),
        "Timișoara": (45.75, 21.23),
        "Iași": (47.16, 27.58),
        "Constanța": (44.18, 28.63),
        "Brașov": (45.65, 25.60),
        "Pitești": (44.85, 24.87)
    }

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        city = st.selectbox("Alege Orașul:", list(cities.keys()))
    with col_btn:
        st.write("")  # Spacer
        run_btn = st.button("Actualizează Datele", type="primary")

    if run_btn:
        with st.spinner(f"Conectare la stația meteo {city}..."):
            lat, lon = cities[city]

            # 1. Get Live Data (Last 24h)
            hist_df = get_live_data(lat, lon)

            # 2. Prep Input for AI
            # Convert timestamp to Sin/Cos features
            time_feats_list = [calculate_time_features(ts) for ts in hist_df['timestamp']]
            time_df = pd.DataFrame(time_feats_list, columns=['day_sin', 'day_cos', 'year_sin', 'year_cos'])

            # Combine Physical + Time features (9 cols)
            full_input = pd.concat([hist_df[config.TARGET_COLS], time_df], axis=1)

            # Scale
            input_scaled = scaler.transform(full_input)

            # 3. Forecast
            forecast_df = forecast_next_24h(model, scaler, input_scaled)

            # 4. Display
            display_results(hist_df, forecast_df, city)


def page_manual_sim(model, scaler):
    st.header("🎛️ Simulator Scenarii (Stress Test)")
    st.markdown("Creează un scenariu manual pentru a testa reacția rețelei neuronale la condiții extreme.")

    with st.form("sim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.number_input("Temp (°C)", -30.0, 50.0, 25.0)
            hum = st.number_input("Umiditate (%)", 0.0, 100.0, 50.0)
        with col2:
            pres = st.number_input("Presiune (hPa)", 900.0, 1050.0, 1013.0)
            wind = st.number_input("Vânt (m/s)", 0.0, 50.0, 5.0)
        with col3:
            rain = st.number_input("Ploaie (mm)", 0.0, 100.0, 0.0)
            sim_time = st.time_input("Ora Simulării", datetime.now().time())
            sim_date = st.date_input("Data Simulării", datetime.now().date())

        submitted = st.form_submit_button("Generează Prognoză 24h")

    if submitted:
        with st.spinner("Se rulează simularea..."):
            # 1. Create Fake History (Last 24h constant based on input)
            # This simulates a "stable day" suddenly changing, or consistently bad weather
            current_dt = datetime.combine(sim_date, sim_time)
            timestamps = [current_dt - timedelta(hours=i) for i in range(24)][::-1]

            # Feature Vectors
            t_feats_list = [calculate_time_features(ts) for ts in timestamps]
            phy_feats = [temp, hum, pres, wind, rain]

            # Construct Sequence (24, 9)
            seq_data = []
            for t_feat in t_feats_list:
                seq_data.append(phy_feats + t_feat)

            input_df = pd.DataFrame(seq_data, columns=config.FEATURE_COLS)

            # Scale
            input_scaled = scaler.transform(input_df)

            # 2. Forecast
            forecast_df = forecast_next_24h(model, scaler, input_scaled)

            # 3. Create a fake "Current Conditions" DF for the display function
            current_cond = pd.DataFrame([{
                'temperature': temp, 'humidity': hum, 'pressure': pres,
                'wind_speed': wind, 'precipitation': rain
            }])

            # 4. Display
            display_results(current_cond, forecast_df, "Scenariu Simulat")


def page_esp32_monitor():
    st.header("📡 ESP32 Live Monitor")
    st.info("Această pagină va afișa datele primite în timp real de la microcontroler.")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/espressif/arduino-esp32/master/docs/esp32_devkitc_v4.png",
                 caption="ESP32 DevKit", width=200)
    with col2:
        st.markdown("### Status Conexiune")
        st.warning("⚠️ Dispozitiv Deconectat")
        st.code("Waiting for serial data on COM3...", language="bash")

    st.markdown("### Date Senzori (Mockup)")
    st.dataframe(pd.DataFrame({
        "Sensor": ["DHT22 (Temp)", "DHT22 (Hum)", "BMP180 (Press)"],
        "Value": ["--", "--", "--"],
        "Last Update": ["Never", "Never", "Never"]
    }), use_container_width=True)


# --- MAIN APP ---
def main():
    model, scaler = load_ai_core()

    tab1, tab2, tab3 = st.tabs(["🇷🇴 România Live", "🎛️ Simulator", "📡 ESP32 Monitor"])

    with tab1:
        page_romania_live(model, scaler)
    with tab2:
        page_manual_sim(model, scaler)
    with tab3:
        page_esp32_monitor()


if __name__ == "__main__":
    main()