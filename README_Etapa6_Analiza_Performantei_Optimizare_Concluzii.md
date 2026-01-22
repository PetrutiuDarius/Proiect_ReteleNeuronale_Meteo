# 📘 README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Petruțiu Darius-Simion  
**Link Repository GitHub:** https://github.com/PetrutiuDarius/Proiect_ReteleNeuronale_Meteo.git  
**Data predării:** 15.01.2026  

---

## Scopul Etapei 6

Această etapă reprezintă punctul culminant al proiectului, concentrându-se pe **validarea științifică** a soluției propuse. Documentul detaliază experimentele de optimizare care au transformat un model de bază ("Baseline") într-un sistem performant ("Production-Ready"), analiza comparativă a rezultatelor și integrarea finală într-o aplicație software complexă.

**Obiectiv principal:** Demonstrarea superiorității arhitecturii cu **Time Embeddings (9 Features)** față de abordarea clasică (5 Features) și livrarea unui Dashboard funcțional.

---

## 1. Experimente de Optimizare și Evoluție

Pe parcursul dezvoltării, au fost efectuate 3 iterații majore pentru a îmbunătăți performanța rețelei LSTM.

### Tabel Centralizator Experimente

| Experiment | Descriere Modificare | Justificare Tehnica | Impact Observat |
| :--- | :--- | :--- | :--- |
| **v1.0 (Baseline)** | 5 Features Fizice (Temp, Hum, Pres, Wind, Rain). | Abordarea standard "Raw Data". | Modelul nu distingea ciclul zi/noapte. Erori mari la Vânt ($R^2 \approx 0.3$). |
| **v1.1 (Data Augmentation)** | Calibrare date sintetice "Black Swan". | Reducerea intensității ploii sintetice (de la 50mm la 15mm/h). | Eliminarea "halucinațiilor" de ploi torențiale. $R^2$ la ploaie a devenit pozitiv. |
| **v2.0 (Optimized)** | **Adăugare 4 Time Embeddings** (Sin/Cos Day/Year). | Introducerea ciclicității matematice. LSTM știe acum ora și anotimpul. | **Creștere masivă:** Temp $R^2 \rightarrow 0.98$, Vânt $R^2 \rightarrow 0.67$. |
| **v2.1 (Physics-Informed)** | Post-procesare cu constrângeri fizice. | Corecția ieșirilor imposibile (ex: ploaie negativă). | Grafice curate, eliminarea zgomotului de fond (<0.1mm). |

---

## 2. Analiza Comparativă: Baseline (5 Features) vs. Optimizat (9 Features)

Analiza se bazează pe fișierele salvate în `results/` și graficele din `docs/`.

### 2.1 Metrici de Regresie (Test Set 2024)

Modelul optimizat (cu 9 intrări) surclasează modelul inițial la toate categoriile, demonstrând importanța contextului temporal în seriile de timp.

| Parametru | Metrica | Model V1 (5 Features) | Model V2 (9 Features - Final) | Îmbunătățire |
| :--- | :--- | :--- | :--- | :--- |
| **Temperatură** | **R2 Score** | 0.9530 | **0.9847** | 🔺 +3.3% |
| | **MAE** | 1.53 °C | **0.88 °C** | 📉 -42% (Eroare redusă) |
| **Vânt** | **R2 Score** | 0.3332 | **0.6734** | 🚀 **+102% (Dublare)** |
| **Umiditate** | **R2 Score** | 0.7968 | **0.9301** | 🔺 +16.7% |
| **Precipitații** | **MAE** | 0.19 mm | **0.07 mm** | 📉 -63% (Precizie chirurgicală) |

**Interpretare:**
* **Vântul:** Saltul de la 0.33 la 0.67 demonstrează că vântul are o componentă puternic dependentă de momentul zilei (brize termice), pe care modelul V1 nu o putea capta.
* **Temperatura:** Scăderea erorii sub 1 grad (0.88°C) face modelul viabil comercial.

### 2.2 Analiza Vizuală (Grafice Comparative)

Comparând graficele generate în `docs/`, se observă stabilitatea superioară a modelului final.

* **Grafic V1 (5 inputs):** [prediction_plot_5_input_parameters.png](docs/prediction_plot_5_input_parameters.png) - Liniile de predicție au "zgomot" și ratează vârfurile locale.
* **Grafic V2 (9 inputs):** [prediction_plot.png](docs/prediction_plot.png) - Linia roșie (AI) se suprapune aproape perfect peste cea albastră (Real), mai ales la temperatură și presiune.

---

## 3. Integrarea în Aplicația Software (Produs Final)

Proiectul a evoluat de la scripturi izolate la un ecosistem software complet.

### Componente Implementate în Etapa 6:
1.  **Dashboard Interactiv (`src/app/dashboard.py`):**
    * Interfață Web (Streamlit) cu 3 module: Live România, Simulator Manual, Monitor ESP32.
    * Vizualizare tabele orare și grafice interactive (Plotly).
    * **Sistem de Alerte:** Detectează automat condiții de Caniculă, Furtună (scădere presiune) sau Îngheț.
2.  **Pipeline Orchestrator (`main.py`):**
    * Sistem inteligent care verifică integritatea datelor și a modelelor.
    * Permite rularea "One-Click" (`python main.py`), gestionând automat descărcarea, generarea sintetică, antrenarea și evaluarea.
3.  **Logica "Physics Constraints":**
    * Implementată în `evaluate.py` și `dashboard.py`.
    * Filtrează aberațiile (ex: Umiditate > 100%, Ploaie < 0).

---

## 4. Analiza Erorilor și Limitări

Chiar și modelul optimizat prezintă limitări inerente naturii haotice a vremii:

1.  **Predicția Vântului la Rafală:** Deși R2 a crescut la 0.67, modelul tinde să subestimeze rafalele extreme (ex: 25 m/s). *Cauză:* LSTM învață media, iar rafalele sunt adesea outliers statistici.
2.  **Ploaia de tip "Aversă Locală":** Modelul prezice probabilitatea condițiilor de ploaie, dar nu poate localiza exact norul deasupra senzorului.
3.  **Dependența de Istoric:** Dacă senzorul ESP32 se defectează și trimite date eronate, modelul va propaga eroarea timp de 24 de ore (garbage in, garbage out).

---

## 5. Concluzii Finale

Proiectul **SIA-Meteo** a atins și depășit obiectivele inițiale, demonstrând aplicabilitatea Rețelelor Neuronale Recurente în meteorologie.

**Puncte Forte (Key Achievements):**
* [x] **Arhitectură Hibridă:** Utilizarea datelor sintetice a permis modelului să învețe scenarii de catastrofă absente din istoricul recent.
* [x] **Precizie Ridicată:** MAE < 0.9°C la temperatură este competitiv cu stațiile meteo comerciale.
* [x] **Inginerie Robustă:** Implementarea Time Embeddings (Sin/Cos) a fost factorul decisiv în optimizare.
* [x] **Aplicabilitate:** Dashboard-ul permite utilizarea imediată atât pentru monitorizare urbană, cât și pentru agricultură (alerte îngheț).

**Direcții Viitoare:**
* Integrarea fizică a senzorului ESP32 (codul de monitorizare există deja în dashboard).
* Implementarea unei arhitecturi Transformer (ex: Temporal Fusion Transformer) pentru a depăși limitele LSTM pe secvențe foarte lungi.

---

## 6. Structura Finală a Repository-ului

```text
Proiect_ReteleNeuronale_Meteo/
├── config/
│   └── preprocessing_params.pkl       # Scaler Antrenat (9 features)
├── data/
│   ├── generated/                     # Dataset Hibrid
│   └── ... (train/val/test splits)
├── docs/
│   ├── loss_curve.png                 # Grafic convergență V2
│   ├── loss_curve_5_input...png       # Grafic convergență V1 (Istoric)
│   ├── prediction_plot.png            # Performanță V2 (Optim)
│   ├── prediction_plot_5_input...png  # Performanță V1 (Baseline)
│   └── screenshots/                   # Capturi din Dashboard
├── models/
│   ├── trained_model.keras            # Model Final (9 inputs)
│   └── trained_model_5_input...keras  # Model Vechi (5 inputs)
├── results/
│   ├── test_metrics.json              # Rezultate V2
│   └── test_metrics_5_input...json    # Rezultate V1
├── src/
│   ├── app/                           # Interfața Web
│   │   └── dashboard.py
│   ├── data_acquisition/              # ETL & Synthetic Gen
│   ├── neural_network/                # Arhitectura LSTM & Training
│   └── processing/                    # Split & Scaling
├── main.py                            # Orchestrator
└── README_*.md                        # Documentație completă