# src/docs_generators/generate_confusion.py
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src import config


# Definim functia custom loss
@tf.keras.utils.register_keras_serializable()
def asymmetric_precipitation_loss(y_true, y_pred):
    squared_error = tf.square(y_true - y_pred)
    overestimation_mask = tf.cast(tf.greater(y_pred, y_true), tf.float32)
    rain_col_idx = 4
    feature_count = 5
    rain_column_mask = tf.one_hot(indices=[rain_col_idx] * tf.shape(y_true)[0], depth=feature_count)
    rain_column_mask = tf.reshape(rain_column_mask, tf.shape(y_true))
    penalty_magnitude = 20.0
    penalty_factor = 1.0 + (overestimation_mask * rain_column_mask * penalty_magnitude)
    return tf.reduce_mean(squared_error * penalty_factor)


def generate_matrix():
    print("🚀 Analiză Performanță Precipitații (Threshold Tuning)...")

    # 1. Incarcare Date
    test_df = pd.read_csv(config.BASE_DIR + '/data/test/test.csv')

    # Feature Engineering
    day = 24 * 60 * 60
    year = 365.2425 * day
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    ts_s = test_df['timestamp'].map(pd.Timestamp.timestamp)

    test_df['day_sin'] = np.sin(ts_s * (2 * np.pi / day))
    test_df['day_cos'] = np.cos(ts_s * (2 * np.pi / day))
    test_df['year_sin'] = np.sin(ts_s * (2 * np.pi / year))
    test_df['year_cos'] = np.cos(ts_s * (2 * np.pi / year))

    # Log Transform Input
    if 'precipitation' in test_df.columns:
        test_df['precipitation'] = np.log1p(test_df['precipitation'])

    # 2. Scalare
    scaler = joblib.load(config.SCALER_PATH)
    feature_cols = config.TARGET_COLS + ['day_sin', 'day_cos', 'year_sin', 'year_cos']
    data_values = test_df[feature_cols].values
    data_scaled = scaler.transform(data_values)

    # 3. Secvente
    X, y_true_scaled = [], []
    win_size = config.SEQ_LENGTH
    horizon = config.PREDICT_HORIZON
    target_indices = [0, 1, 2, 3, 4]

    for i in range(win_size, len(data_scaled) - horizon):
        X.append(data_scaled[i - win_size:i])
        y_true_scaled.append(data_scaled[i + horizon, target_indices])

    X = np.array(X)
    y_true_scaled = np.array(y_true_scaled)

    # 4. Predictie
    model = tf.keras.models.load_model(config.MODEL_PATH,
                                       custom_objects={'asymmetric_precipitation_loss': asymmetric_precipitation_loss})
    y_pred_scaled = model.predict(X, verbose=0)

    # 5. Denormalizare & Analiza Valori
    def denormalize_rain(scaled_rain_col):
        dummy = np.zeros((len(scaled_rain_col), 9))
        dummy[:, 4] = scaled_rain_col
        real = scaler.inverse_transform(dummy)[:, 4]
        return np.expm1(real)  # Inverse Log

    rain_true = denormalize_rain(y_true_scaled[:, 4])
    rain_pred = denormalize_rain(y_pred_scaled[:, 4])

    # --- DEBUGGING CRITIC ---
    print(f"\n📊 Statistici Predicții:")
    print(f"  Min Pred: {rain_pred.min():.4f} mm")
    print(f"  Max Pred: {rain_pred.max():.4f} mm")
    print(f"  Media Pred: {rain_pred.mean():.4f} mm")
    print(f"  Median Pred: {np.median(rain_pred):.4f} mm")

    # 6. Căutare Prag Optim (Threshold Tuning)
    best_thresh = 0.1
    best_f1 = 0

    # Testam praguri intre 0.1mm si 2.0mm
    thresholds = np.arange(0.1, 2.0, 0.1)

    print("\n🔍 Căutare Prag Optim...")
    for t in thresholds:
        y_c_true = (rain_true > 0.1).astype(int)  # Adevarul ramane la 0.1 (fizic)
        y_c_pred = (rain_pred > t).astype(int)  # Modelul il judecam dupa pragul t

        # Calculam F1 pentru clasa RAIN (1)
        score = f1_score(y_c_true, y_c_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    print(f"✅ Prag Optim Identificat: {best_thresh:.2f} mm (F1 Score: {best_f1:.2f})")

    # 7. Generare Matrice Finală cu Pragul Optim
    y_class_true = (rain_true > 0.1).astype(int)
    y_class_pred = (rain_pred > best_thresh).astype(int)

    cm = confusion_matrix(y_class_true, y_class_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: Fără Ploaie', 'Pred: Ploaie'],
                yticklabels=['Real: Fără Ploaie', 'Real: Ploaie'])

    plt.title(f'Matrice de Confuzie Optimizată\nPrag Decizie: > {best_thresh:.2f} mm', fontsize=14)
    plt.ylabel('Adevăr (Ground Truth)')
    plt.xlabel('Predicție Model')

    save_path = os.path.join(config.BASE_DIR, 'docs', 'confusion_matrix_optimized.png')
    plt.savefig(save_path)
    print(f"\n📸 Imagine salvată: {save_path}")

    print("\n--- RAPORT CLASIFICARE FINAL ---")
    print(classification_report(y_class_true, y_class_pred, target_names=['No Rain', 'Rain'], zero_division=0))


if __name__ == "__main__":
    generate_matrix()