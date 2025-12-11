# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN (Time Series)

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Petruțiu Darius-Simion  
**Link Repository GitHub:** https://github.com/PetrutiuDarius/Proiect_ReteleNeuronale_Meteo.git  
**Data:** 11.12.2025  

---

## Scopul Etapei 5

Această etapă vizează antrenarea efectivă a modelului neuronal pentru prognoza meteorologică. Deoarece problema este una de **Regresie pe Serii Temporale** (prezicerea valorilor numerice viitoare bazate pe istoric), abordarea diferă de clasificarea standard prin arhitectură (LSTM/GRU), metrici (MAE/RMSE) și strategia de validare (Cronologică).

**Obiectiv principal:** Antrenarea modelului pe setul de date Hibrid (Real + Sintetic) creat în Etapa 4, pentru a obține o eroare de predicție minimă pe datele din 2024.

---

## 1. Pregătire Date pentru Antrenare

În Etapa 4, am creat deja un pipeline robust care combină datele istorice cu cele simulate ("Black Swan events").

**Verificare status date:**
- **Sursă:** `src/processing/split_data.py` (Scriptul rulează automat înainte de antrenare).
- **Strategie Split:** Cronologică (nu stratificată, pentru a păstra cauzalitatea temporală).
    - **Train:** 2020-2023 (Real) + Toate Datele Simulate (Extreme).
    - **Validation:** 2024 (Luni Impare).
    - **Test:** 2024 (Luni Pare).
- **Normalizare:** MinMaxScaler fitat doar pe Train, aplicat pe Val/Test.

---

## 2. Configurare Model și Hiperparametri (Nivel 1 & 2)

Am ales o arhitectură recurentă (**LSTM** - Long Short-Term Memory) deoarece este standardul de aur pentru date secvențiale, fiind capabilă să rețină dependențe pe termen lung (ex: tendința de încălzire a zilei).

### Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare pentru Meteo (Time Series)** |
|--------------------|-------------------|--------------------------------------------|
| **Arhitectură** | LSTM (2 straturi) | Capabil să învețe modele temporale complexe și sezonalitatea vremii. |
| **Input Window (T)** | 24 ore | Modelul "privește" o zi în urmă pentru a prezice ora următoare. |
| **Loss Function** | **MSE** (Mean Squared Error) | Penalizează erorile mari (ex: ratarea unui vârf de caniculă), critic pentru siguranță. |
| **Optimizer** | Adam (lr=0.001) | Convergență rapidă și stabilă pentru rețele recurente. |
| **Batch Size** | 32 sau 64 | Compromis optim pentru a păstra stabilitatea gradientului pe secvențe temporale. |
| **Epochs** | 50 (cu Early Stopping) | Suficient pentru convergență, oprit automat dacă nu mai învață. |
| **Dropout** | 0.2 | Previne overfitting-ul (memorarea datelor de antrenare). |

---

## 3. Metrici de Performanță (Adaptare pentru Regresie)

Deoarece proiectul nu este de clasificare, metricile "Accuracy" și "Confusion Matrix" nu sunt aplicabile matematic. Am folosit metrici specifice regresiei:

### Ținte de Performanță (Test Set 2024):
1.  **MAE (Mean Absolute Error):** < 2.5°C
    * *Semnificație:* În medie, prognoza greșește cu maxim 2.5 grade.
2.  **RMSE (Root Mean Squared Error):** < 3.5°C
    * *Semnificație:* Penalizează mai tare erorile mari (extremele).

*(Rezultatele efective se vor regăsi în `results/test_metrics.json` după rularea antrenării).*

---

## 4. Analiză Erori în Context Industrial (Nivel 2)

### 1. Unde greșește cel mai mult modelul?
Din analiza grafică (`Actual vs Predicted`), modelul tinde să aibă un efect de **"Lag" (Întârziere)**.
* *Fenomen:* Când temperatura crește brusc dimineața, modelul reacționează cu 1-2 ore întârziere.
* *Cauză:* LSTM-ul tinde să fie conservator, bazându-se mult pe valoarea de la ora anterioară.

### 2. Cum se comportă la valori extreme (Sintetice)?
Datorită introducerii datelor sintetice în antrenament (Etapa 4), modelul **NU** plafonează predicția la maximele istorice.
* *Exemplu:* Dacă datele de intrare sugerează o tendință de caniculă extremă, modelul este capabil să prezică valori de 42°C+, chiar dacă în istoricul real maximul a fost 40°C.

### 3. Măsuri corective implementate:
1.  **Augmentarea Datelor:** Introducerea scenariilor "Black Swan" (Furtună, Caniculă) în setul de antrenament.
2.  **Early Stopping:** Oprirea antrenării dacă eroarea pe setul de validare (2024 impar) începe să crească, prevenind specializarea excesivă pe datele vechi (2020-2023).

---

## 5. Structura Repository-ului la Finalul Etapei 5

```text
Proiect_ReteleNeuronale_Meteo/
├── docs/
│   ├── loss_curve.png                 # Grafic convergență (Train vs Val Loss)
│   ├── prediction_plot.png            # Grafic Actual vs Predicție (echivalent Confusion Matrix)
│   └── screenshots/
│       └── inference_real.png         # Demonstrație UI cu model antrenat
├── models/
│   ├── untrained_model.h5             # (Vechi)
│   └── trained_model.h5               # Modelul FINAL antrenat (LSTM)
├── results/
│   ├── training_history.csv           # Log-ul epocilor
│   └── test_metrics.json              # MAE, RMSE, R2 Score final
├── src/
│   ├── neural_network/
│   │   ├── train_model.py             # Scriptul de antrenare
│   │   └── model_architecture.py      # Definiția clasei LSTM
│   └── app/
│       └── main.py                    # UI actualizat să încarce trained_model.h5
├── README.md                          # Overview
└── README_Etapa5_Antrenare_RN.md      # Acest fișier