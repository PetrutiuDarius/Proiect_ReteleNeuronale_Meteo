# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date

**Disciplina:** Rețele Neuronale <br />
**Instituție:** POLITEHNICA București – FIIR <br />
**Student:** Petruțiu Darius-Simion <br />
**Link Repository GitHub:** https://github.com/PetrutiuDarius/Proiect_ReteleNeuronale_Meteo.git <br />
**Data:** 04.11.2025 <br />

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care s-a analizat și preprocesat setul de date necesar proiectului „Prognoza Meteo”. Scopul etapei a fost transformarea datelor meteorologice brute într-un format optim pentru rețele neuronale (serii temporale normalizate), rezolvând totodată problema lipsei de fenomene extreme din datele istorice.

---

## 1. Structura Repository-ului Github (Versiunea Etapei 3)

```text
Proiect_ReteleNeuronale_Meteo/
├── README.md
├── etapa3_analiza_date.md         # Acest fișier
├── data/
│   ├── raw/                       # Date brute (istoric Open-Meteo)
│   ├── generated/                 # Date sintetice (Extreme) și Dataset Hibrid
│   ├── train/                     # Set de instruire (2020-2023 + Sintetic)
│   ├── validation/                # Set de validare (2024 Luni Impare)
│   └── test/                      # Set de testare (2024 Luni Pare)
├── src/
│   ├── data_acquisition/          # Scripturi descărcare și generare date
│   └── processing/                # Scripturi de split și normalizare
├── requirements.txt               # Dependențe Python
```

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor
* **Origine:** API Open-Meteo (Historical Weather) + Generator Sintetic Propriu.
* **Modul de achiziție:**
  * Descarcă date reale (API Request) via `src/data_acquisition/data_loader.py`.
  * Generare programatică (Algoritm statistic) via `src/data_acquisition/synthetic_generator.py`.
* **Perioada:** 01.01.2020 – 31.12.2024.

### 2.2 Caracteristicile dataset-ului
* **Număr total de observații:** ~60,000 ore (din care ~25,000 simulate).
* **Număr de caracteristici (features):** 4 (Temperatură, Umiditate, Presiune, Vânt).
* **Tipuri de date:** Numerice (Serii Temporale Multivariate).
* **Format fișiere:** CSV.

### 2.3 Descrierea fiecărei caracteristici

| Caracteristică | Tip | Unitate | Descriere | Domeniu valori (Real+Simulat) |
|---|---|---|---|---|
| temperature | numeric | °C | Temperatura aerului la 2m | -15.0 ... +44.0 |
| humidity | numeric | % | Umiditatea relativă | 20.0 ... 100.0 |
| pressure | numeric | hPa | Presiunea atmosferică | 980.0 ... 1030.0 |
| wind_speed | numeric | m/s | Viteza vântului la 10m | 0.0 ... 30.0 |

## 3. Analiza Exploratorie a Datelor (EDA)

### 3.1 Statistici descriptive aplicate
* S-a analizat distribuția datelor reale pe perioada 2020-2024.
* **Concluzie:** Datele reale au o distribuție normală, dar lipsesc valorile extreme critice pentru siguranță (ex: nu există temperaturi > 42°C sau vânt > 25m/s în istoric).

### 3.2 Analiza calității datelor
* **Valori lipsă:** Open-Meteo furnizează date complete. Eventualele goluri minore sunt tratate prin interpolare liniară (`method='time'`).
* **Consistență:** S-a verificat cronologia timestamp-urilor.

### 3.3 Probleme identificate
* **Problemă:** "Imbalanced Dataset" în ceea ce privește fenomenele extreme. Evenimentele de tip "Furtună violentă" sau "Caniculă extremă" reprezentau < 0.1% din datele reale.
* **Soluție:** Augmentarea setului de date prin generarea a 25,000 de ore de date sintetice ("Black Swan events") care au fost adăugate la setul de antrenare.

## 4. Preprocesarea Datelor

### 4.1 Curățarea și Transformarea
Procesul este automatizat în `src/processing/split_data.py`:
* **Imputare:** Interpolare liniară pentru continuitate temporală.
* **Normalizare:** S-a utilizat **MinMax Scaler** pentru a aduce toate valorile în intervalul `[0, 1]`.
* **Notă:** Scalerul a fost antrenat (`.fit`) **DOAR** pe setul de antrenare pentru a evita *Data Leakage*, și apoi aplicat pe validare și test.

### 4.2 Structurarea seturilor de date
S-a ales o împărțire cronologică modificată (nu aleatorie/stratificată), specifică seriilor temporale:
* **Train (70%):** Anii 2020-2023 (Real) + Toate Datele Simulate.
* **Validation (15%):** Anul 2024 (Luni Impare: Ian, Mar, ...).
* **Test (15%):** Anul 2024 (Luni Pare: Feb, Apr, ...).

### 4.3 Salvarea rezultatelor
* Fișierele CSV finale sunt salvate în `data/train/`, `data/validation/`, `data/test/`.
* Obiectul Scaler este salvat în `data/scalers/minmax_scaler.pkl` pentru a putea fi folosit ulterior la denormalizarea predicțiilor.

## 5. Stare Etapă
- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date generate sintetic (rezolvare lipsă extreme)
- [x] Date preprocesate și normalizate
- [x] Seturi train/validation/test generate
- [x] Documentație actualizată