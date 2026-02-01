# src/app/dashboard.py
"""
SIA-Meteo AI Dashboard - Main UI Application.

This module implements the user interface for the meteorological monitoring and forecasting system.
It integrates data visualization (Plotly), AI inference (TensorFlow/Keras), and real-time
IoT connectivity (Azure IoT Hub).

Key Features:
1. **Live Monitoring:** Displays real-time data from ESP32 sensors via Azure IoT Hub.
2. **Adaptive AI:** Enables on-demand retraining of the neural network for new geographic locations.
3. **Forecasting:** Provides 24-hour weather predictions using LSTM neural networks.
4. **Scenario simulation:** Allows manual input testing ("What-If" scenarios).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import os
import requests
import sys
import tensorflow as tf
import time
import json
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from geopy.geocoders import Nominatim

# Ensures project root is in the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import config
from src.app.adaptive_training import train_adaptive_model

# =============================================================================
#  UI CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="SIA-Meteo AI Dashboard",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced visual hierarchy and alert boxes
st.markdown("""
    <style>
    /* Metric Cards Styling */
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    /* Alert Boxes Styling */
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .alert-critical {
        background-color: #ffcccc;
        color: #990000;
        border: 1px solid #990000;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #856404;
    }
    /* Progress Bar Color Override */
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
#  CORE AI FUNCTIONS (LOSS & LOADING)
# =============================================================================

@tf.keras.utils.register_keras_serializable()
def asymmetric_precipitation_loss(y_true, y_pred):
    """
    Custom Loss Function for precipitation forecasting.

    Implements a penalty mechanism that heavily penalizes underestimation of rainfall events,
    while being more lenient on false positives. This addresses the class imbalance issue
    where rain events are rare compared to dry periods.
    """
    squared_error = tf.square(y_true - y_pred)
    overestimation_mask = tf.cast(tf.greater(y_pred, y_true), tf.float32)

    # Identify the precipitation column index (assumed to be 4 based on config)
    rain_col_idx = 4
    feature_count = 5
    rain_column_mask = tf.one_hot(indices=[rain_col_idx] * tf.shape(y_true)[0], depth=feature_count)
    rain_column_mask = tf.reshape(rain_column_mask, tf.shape(y_true))

    penalty_magnitude = 20.0
    # Apply penalty only where overestimation occurs in the rain column
    penalty_factor = 1.0 + (overestimation_mask * rain_column_mask * penalty_magnitude)
    return tf.reduce_mean(squared_error * penalty_factor)


@st.cache_resource
def load_ai_core():
    """
    Loads the default pre-trained model and scaler into memory.
    Uses st.cache_resource to prevent reloading on every interaction.
    """
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        st.error("🚨 Critical Error: Model or Scaler not found. Please run 'main.py' first.")
        st.stop()

    try:
        model = load_model(
            config.MODEL_PATH,
            custom_objects={'asymmetric_precipitation_loss': asymmetric_precipitation_loss}
        )
        scaler = joblib.load(config.SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load default AI core: {e}")
        st.stop()


def load_local_ai(folder_path):
    """
    Loads a location-specific model and scaler for adaptive inference.

    Args:
        folder_path (str): Directory containing the custom model artifacts.

    Returns:
        tuple: (model, scaler) or (None, None) if loading fails.
    """
    m_path = os.path.join(folder_path, "model.keras")
    s_path = os.path.join(folder_path, "scaler.pkl")

    try:
        model = load_model(m_path, custom_objects={'asymmetric_precipitation_loss': asymmetric_precipitation_loss})
        scaler = joblib.load(s_path)
        return model, scaler
    except Exception as e:
        st.error(f"Could not load local model: {e}")
        return None, None

# =============================================================================
#  DATA PROCESSING & UTILITIES
# =============================================================================

def calculate_time_features(timestamp):
    """
    Converts a timestamp into cyclical Sin/Cos features for Neural Network input.
    Captures daily (24h) and annual (365 days) seasonality patterns.
    """
    day = 24 * 60 * 60
    year = 365.2425 * day
    ts_s = timestamp.timestamp()
    return [
        np.sin(ts_s * (2 * np.pi / day)), np.cos(ts_s * (2 * np.pi / day)),
        np.sin(ts_s * (2 * np.pi / year)), np.cos(ts_s * (2 * np.pi / year))
    ]

@st.cache_data(ttl=3600)
def get_location_name(lat, lon):
    """
    Performs reverse geocoding to retrieve city name from coordinates.
    Cached for 1 hour to respect API rate limits.
    """
    try:
        geolocator = Nominatim(user_agent="sia_meteo_app")
        location = geolocator.reverse((lat, lon), language='ro')
        if location:
            address = location.raw['address']
            city = address.get('city', address.get('town', address.get('village', 'Locație necunoscută')))
            county = address.get('county', '')
            return f"{city}, {county}"
        return "Locație necunoscută"
    except:
        return "Eroare Geolocație"


def get_live_data(lat, lon):
    """
    Fetches real-time weather history (last 24h) from Open-Meteo API.
    Used for the 'Romania Live' page to initialize the LSTM sequence.
    """
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&past_days=1&forecast_days=1&hourly=temperature_2m,relative_humidity_2m,"
           f"surface_pressure,wind_speed_10m,precipitation&timezone=auto")
    try:
        r = requests.get(url, timeout=5).json()
        data = {
            'timestamp': pd.to_datetime(r['hourly']['time']),
            'temperature': r['hourly']['temperature_2m'],
            'humidity': r['hourly']['relative_humidity_2m'],
            'pressure': r['hourly']['surface_pressure'],
            'wind_speed': r['hourly']['wind_speed_10m'],
            'precipitation': r['hourly']['precipitation']
        }
        df = pd.DataFrame(data)
        df['wind_speed'] = df['wind_speed'].clip(upper=8.0)

        # Filter strictly for the last 24 hours relative to now
        current_time = pd.Timestamp.now().floor('h')
        df = df[df['timestamp'] <= current_time].tail(24)
        return df.reset_index(drop=True)
    except:
        return pd.DataFrame()  # Return empty DF on failure

# =============================================================================
#  FORECASTING LOGIC
# =============================================================================

def forecast_next_24h(model, scaler, initial_sequence_24h, start_time):
    """
    Generates a 24-hour hour-by-hour forecast using autoregression.

    The prediction at t+1 is fed back into the input sequence to predict t+2,
    allowing long-term forecasting from a single trained model.
    """
    predictions = []
    current_seq = initial_sequence_24h.copy()
    last_timestamp = start_time

    for i in range(24):
        # 1. Predict the next step
        input_tensor = np.array([current_seq])
        pred_scaled_5 = model.predict(input_tensor, verbose=0)[0]

        # The scaler expects 0-1 usually, but outliers can go slightly outside.
        # Extreme values like -10 or +10 indicate model instability.
        # I clamp conservatively to stop explosion.
        pred_scaled_5 = np.clip(pred_scaled_5, -0.5, 1.5)

        # 2. Denormalize prediction
        dummy = np.zeros((1, 9))  # Create a dummy array to match scaler's expected shape
        dummy[:, :5] = pred_scaled_5
        pred_real_5 = scaler.inverse_transform(dummy)[0, :5]

        # 3. Inverse Log Transform for Precipitation
        pred_real_5[4] = np.expm1(pred_real_5[4])

        # 4. Apply physical constraints
        temp = pred_real_5[0]
        hum = min(max(pred_real_5[1], 0), 100)  # Clamp humidity 0-100%
        pres = pred_real_5[2]
        wind = min(max(pred_real_5[3], 0), 8)  # Wind cannot be negative and beyond 8
        rain = max(pred_real_5[4], 0)  # Rain cannot be negative

        # Apply a noise gate to filter micro-values
        if rain < 0.1: rain = 0.0

        # Determine Weather Condition (Rain/Snow/Clear)
        is_snow = (rain > 0) and (temp <= config.SNOW_TEMP_THRESHOLD)
        precip_type = "❄️ Ninsoare" if is_snow else ("🌧️ Ploaie" if rain > 0 else "☁️ Noros/Senin")

        next_time = last_timestamp + timedelta(hours=i + 1)
        predictions.append({
            'Ora': next_time.strftime('%H:%M'),
            'Temp (°C)': round(temp, 1),
            'Umiditate (%)': round(hum, 1),
            'Presiune (hPa)': round(pres, 1),
            'Vânt (m/s)': round(wind, 1),
            'Precipitații (mm)': round(rain, 2),
            'Condiție': precip_type
        })

        # 5. Prepare the sequence for the next iteration (Autoregression loop)
        pred_for_feedback = pred_real_5.copy()
        pred_for_feedback[4] = np.log1p(pred_for_feedback[4])  # Re-apply Log for model input

        new_time_feats = calculate_time_features(next_time)
        row_real_9 = np.concatenate([pred_for_feedback, new_time_feats])

        # Scale the new row and append to the sliding window
        row_scaled_9 = scaler.transform([row_real_9])[0]
        current_seq = np.vstack([current_seq[1:], row_scaled_9])

    return pd.DataFrame(predictions)

def analyze_alerts(df):
    """
    Analyzes the forecast dataframe to identify potential extreme weather events.
    Returns a list of alert messages.
    """
    alerts = []
    max_temp = df['Temp (°C)'].max()
    max_wind = df['Vânt (m/s)'].max()
    max_rain = df['Precipitații (mm)'].max()

    if max_temp > 38.0: alerts.append(("🔥 CANICULĂ EXTREMĂ", f"Temperatura va atinge {max_temp}°C."))
    elif max_temp > 35.0: alerts.append(("🟠 AVERTIZARE CĂLDURĂ", f"Max: {max_temp}°C"))

    if max_wind > 20.0:
        alerts.append(("🌪️ FURTUNĂ VIOLENTĂ", f"Vânt: {max_wind} m/s"))
    elif max_wind > 15.0:
        alerts.append(("💨 VÂNT PUTERNIC", f"Rafale de {max_wind} m/s"))

    if max_rain > 10.0:
        alerts.append(("⛈️ PLOI TORENȚIALE", f"Acumulări de {max_rain} mm/h."))
    elif max_rain > 0.0 and df['Temp (°C)'].min() <= 0.5:
        alerts.append(("❄️ RISC DE ÎNGHEȚ/ZĂPADĂ", "Condiții de polei sau ninsoare."))

    return alerts

# =============================================================================
#  UI COMPONENT: RESULT DISPLAY
# =============================================================================

def display_results(current_conditions, forecast_df, city_name, start_time):
    """Reusable component to show metrics, alerts, charts, and table."""

    st.divider()

    # Header with city and timestamp
    col_header_l, col_header_r = st.columns([2, 1])
    with col_header_l:
        st.subheader(f"📍 Condiții actuale: {city_name}")
    with col_header_r:
        st.info(f"🕒 Referință: **{start_time.strftime('%d-%m-%Y %H:%M')}**")

    # Current metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Temperatură", f"{current_conditions['temperature'].iloc[-1]:.1f} °C")
    c2.metric("Umiditate", f"{current_conditions['humidity'].iloc[-1]:.0f} %")
    c3.metric("Presiune", f"{current_conditions['pressure'].iloc[-1]:.0f} hPa")
    c4.metric("Vânt", f"{current_conditions['wind_speed'].iloc[-1]:.1f} m/s")
    rain_val = current_conditions['precipitation'].iloc[-1]
    c5.metric("Precipitații", f"{rain_val:.1f} mm")

    st.divider()

    # Alerts section
    alerts = analyze_alerts(forecast_df)
    if alerts:
        st.subheader("⚠️ Situații extreme detectate (următoarele 24h)")
        for title, msg in alerts:
            if "CRITICĂ" in title or "VIOLENTĂ" in title or "CANICULĂ" in title:
                st.markdown(f"<div class='alert-box alert-critical'>{title}: {msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='alert-box alert-warning'>{title}: {msg}</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Prognoza arată condiții stabile pentru următoarele 24h.")

    st.divider()

    # 3. Charts and table section
    st.subheader(f"📅 Prognoză detaliată (24 Ore)")

    tab_chart, tab_table = st.tabs(["📉 Grafice evoluție", "📄 Tabel date"])

    with tab_chart:
        # Chart 1: Temp (Line) + Rain (Bar)
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
            title='Evoluție temperatură și precipitații',
            xaxis_title='Ora',
            yaxis=dict(title='Temperatură (°C)', side='left'),
            yaxis2=dict(title='Precipitații (mm)', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Wind + Pressure
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=forecast_df['Ora'], y=forecast_df['Presiune (hPa)'], name='Presiune',
                                  line=dict(color='purple')))
        fig2.add_trace(go.Scatter(x=forecast_df['Ora'], y=forecast_df['Vânt (m/s)'], name='Vânt',
                                  line=dict(color='green', dash='dot'), yaxis='y2'))

        fig2.update_layout(
            title='Evoluție presiune și vânt',
            yaxis=dict(title='Presiune (hPa)'),
            yaxis2=dict(title='Vânt (m/s)', overlaying='y', side='right'),
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Table with all data
    with tab_table:
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

# =============================================================================
#  PAGE IMPLEMENTATIONS
# =============================================================================

def page_romania_live(model, scaler):
    """Page 1: Live monitoring for major Romanian cities via Open-Meteo API."""
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

    col_sel, col_btn = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
    with col_sel:
        city = st.selectbox("Alege orașul:", list(cities.keys()))
    with col_btn:
        run_btn = st.button("Actualizează datele", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Conectare la stația meteo {city}..."):
            lat, lon = cities[city]
            hist_df = get_live_data(lat, lon)

            # Prepare Input Sequence
            start_time = hist_df['timestamp'].iloc[-1]
            time_feats_list = [calculate_time_features(ts) for ts in hist_df['timestamp']]
            time_df = pd.DataFrame(time_feats_list, columns=['day_sin', 'day_cos', 'year_sin', 'year_cos'])

            full_input = pd.concat([hist_df[config.TARGET_COLS], time_df], axis=1)

            # Log Transform Rain
            if 'precipitation' in full_input.columns:
                full_input['precipitation'] = np.log1p(full_input['precipitation'])

            # Inference
            input_scaled = scaler.transform(full_input)
            forecast_df = forecast_next_24h(model, scaler, input_scaled, start_time)

            display_results(hist_df, forecast_df, city, start_time)

def page_manual_sim(model, scaler):
    """Page 2: Manual simulator for testing extreme scenarios."""
    st.header("🎛️ Simulator scenarii")
    st.markdown("Creează un scenariu manual pentru a testa reacția rețelei neuronale.")

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
            sim_time = st.time_input("Ora simulării", datetime.now().time())
            sim_date = st.date_input("Data simulării", datetime.now().date())

        submitted = st.form_submit_button("Generează prognoză 24h", type="primary")

    if submitted:
        with st.spinner("Se rulează simularea..."):
            current_dt = datetime.combine(sim_date, sim_time)

            # Generate synthetic history (repeat current conditions backwards)
            timestamps = [current_dt - timedelta(hours=i) for i in range(24)][::-1]
            t_feats_list = [calculate_time_features(ts) for ts in timestamps]

            rain_log = np.log1p(rain)
            phy_feats = [temp, hum, pres, wind, rain_log]

            seq_data = [phy_feats + t_feat for t_feat in t_feats_list]
            input_df = pd.DataFrame(seq_data, columns=config.FEATURE_COLS)

            # Inference
            input_scaled = scaler.transform(input_df)
            forecast_df = forecast_next_24h(model, scaler, input_scaled, current_dt)

            # Create dummy current conditions for display
            current_cond = pd.DataFrame([{
                'temperature': temp, 'humidity': hum, 'pressure': pres,
                'wind_speed': wind, 'precipitation': rain
            }])

            display_results(current_cond, forecast_df, "Scenariu simulat", current_dt)

def page_esp32_monitor(default_model, default_scaler):
    """Page 3: Real-time IoT Dashboard with Adaptive Training capabilities."""
    st.header("📡 ESP32 Live Monitor & Adaptive AI")
    DATA_FILE = "latest_telemetry.json"

    # Control Panel
    with st.container():
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
        with c1: st.caption("Sistem conectat la Azure IoT Hub via 'azure_listener.py'.")
        with c2:
            if st.button("🔄 Refresh manual", use_container_width=True): st.rerun()
        with c3: auto_refresh = st.toggle("🔴 Auto-Live (5m)", value=False)

    st.divider()

    # 2. Data Loading
    if os.path.exists(DATA_FILE):
        try:
            # Check file update time for toast notification
            file_mod_time = os.path.getmtime(DATA_FILE)
            if 'last_read_time' not in st.session_state: st.session_state['last_read_time'] = 0
            if file_mod_time > st.session_state['last_read_time']:
                st.session_state['last_read_time'] = file_mod_time
                st.toast('🔔 Date noi recepționate!', icon='📡')

            with open(DATA_FILE, 'r') as f:
                data = json.load(f)

            # Metadata Extraction
            device_id = data.get('deviceId', 'Unknown')
            saved_at = data.get('_local_saved_at', 'N/A')
            esp_lat = data.get('lat', 44.43)
            esp_lon = data.get('lon', 26.10)
            location_name = get_location_name(esp_lat, esp_lon)

            # Status Bar
            k1, k2, k3 = st.columns(3)
            k1.metric("Dispozitiv", device_id, "Online")
            k2.metric("Ultimul pachet", saved_at)
            k3.metric("Locație detectată", location_name, f"{esp_lat:.4f}, {esp_lon:.4f}")

            # --- ADAPTIVE AI SECTION ---
            with st.expander("🧠 Administrare model AI (Adaptive Training)", expanded=False):
                st.info("""
                    **Logica de adaptare:** Modelul generic este antrenat pe climatul temperat-continental (București).
                    Dacă senzorul este mutat într-o zonă diferită (ex: munte, mare), re-antrenarea pe date istorice locale
                    este recomandată pentru a îmbunătăți precizia.
                    """)

                c_train_1, c_train_2 = st.columns([2, 1], gap="medium")
                custom_model_dir = f"models/adaptive/{esp_lat}_{esp_lon}"
                has_custom_model = os.path.exists(os.path.join(custom_model_dir, "model.keras"))

                with c_train_1:
                    if has_custom_model:
                        st.success(f"✅ Model optimizat disponibil pentru {location_name}.")
                        try:
                            with open(os.path.join(custom_model_dir, "metrics.json"), 'r') as f:
                                m = json.load(f)
                            st.caption(f"📊 Acuratețe (MAE): **{m['mae']:.4f}** | Antrenat: {m['trained_date']}")
                        except:
                            pass
                    else:
                        st.warning("⚠️ Se utilizează modelul generic. Precizia poate fi afectată de micro-climat.")

                with c_train_2:
                    if st.button("🚀 Antrenează model local", use_container_width=True, help="Durată estimată: 2-5 min"):
                        p_bar = st.progress(0, text="Inițializare...")
                        try:
                            def update_p(msg, val):
                                p_bar.progress(val, text=msg)

                            res = train_adaptive_model(esp_lat, esp_lon, progress_callback=update_p)
                            if "error" in res:
                                st.error(res["error"])
                            else:
                                st.success("Antrenare completă!");
                                time.sleep(1);
                                st.rerun()
                        except Exception as e:
                            st.error(f"Eroare critică: {e}")

                use_custom = st.checkbox("Activează modelul local", value=has_custom_model,
                                         disabled=not has_custom_model)
                st.caption(f"ℹ️ Model activ: **{'Inteligență locală' if use_custom else 'Model generic'}**")

            # 3. Model Selection Logic
            if use_custom:
                active_model, active_scaler = load_local_ai(custom_model_dir)
                if active_model is None: active_model, active_scaler = default_model, default_scaler
            else:
                active_model, active_scaler = default_model, default_scaler

            # 4. Data Processing & Inference
            history = data.get('history', [])
            if history:
                df_esp = pd.DataFrame(history)
                df_esp['timestamp'] = pd.to_datetime(df_esp['timestamp'])

                # Data Healing (Padding missing hours)
                rows_needed = 24 - len(df_esp)
                if rows_needed > 0:
                    base_row = df_esp.iloc[0].copy()
                    new_rows = []
                    for i in range(rows_needed):
                        r = base_row.copy()
                        r['timestamp'] = base_row['timestamp'] - timedelta(hours=i + 1)
                        new_rows.append(r)
                    df_esp = pd.concat([pd.DataFrame(new_rows).sort_values('timestamp'), df_esp]).reset_index(drop=True)

                # Feature Prep
                time_feats = [calculate_time_features(ts) for ts in df_esp['timestamp']]
                time_df = pd.DataFrame(time_feats, columns=['day_sin', 'day_cos', 'year_sin', 'year_cos'])
                model_input = pd.concat([df_esp[config.TARGET_COLS], time_df], axis=1)

                if 'precipitation' in model_input.columns:
                    model_input['precipitation'] = np.log1p(model_input['precipitation'])

                # Predict
                input_scaled = active_scaler.transform(model_input)
                start_time = df_esp['timestamp'].iloc[-1]
                forecast_df = forecast_next_24h(active_model, active_scaler, input_scaled, start_time)

                display_results(df_esp.tail(1), forecast_df, location_name, start_time)
            else:
                st.error("Eroare: Istoric date gol.")

        except Exception as e:
            st.error(f"Eroare dashboard: {e}")
    else:
        st.info("⏳ Se așteaptă prima conexiune de la Azure Listener...")

    # Auto Refresh Logic
    if auto_refresh:
        time.sleep(300)
        st.rerun()

# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def main():
    model, scaler = load_ai_core()

    t1, t2, t3 = st.tabs(["🇷🇴 România Live", "🎛️ Simulator", "📡 ESP32 Monitor"])

    with t1: page_romania_live(model, scaler)
    with t2: page_manual_sim(model, scaler)
    with t3: page_esp32_monitor(model, scaler)

if __name__ == "__main__":
    main()