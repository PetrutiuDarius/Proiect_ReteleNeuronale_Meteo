# Sistem Inteligent pentru Prognoza Meteorologică (SIA-Meteo)

**Student:** Petruțiu Darius-Simion  
**Grupa:** 632AB  
**Disciplina:** Rețele Neuronale  

---

## 1. Descrierea Proiectului
Acest proiect vizează dezvoltarea unui sistem bazat pe Rețele Neuronale (RN) pentru prognoza meteorologică de tip "nowcasting" (termen scurt), optimizat pentru aplicații locale precum managementul energiei regenerabile și agricultura de precizie.

Obiectivul curent (**Etapa 3**) este construirea unui **Pipeline de Date (ETL)** robust, capabil să transforme date istorice brute în seturi de antrenament optimizate pentru învățare automată.

---

## 2. Sursa Datelor
Datele sunt extrase din arhiva istorică **Open-Meteo**, simulând achiziția de la o stație meteo locală amplasată în București.

* **Locație:** București (Lat: 44.4323, Long: 26.1063)
* **Perioada:** 01 Ianuarie 2020 – 31 Decembrie 2024 (5 ani compleți)
* **Rezoluție Temporală:** Orară (Hourly)
* **Variabile de Intrare (Features):**
    * `temperature_2m` (°C)
    * `relative_humidity_2m` (%)
    * `surface_pressure` (hPa)
    * `wind_speed_10m` (m/s)

---

## 3. Arhitectura Pipeline-ului de Date

Procesarea datelor este complet automatizată prin scripturi Python modulare, respectând principii de *Clean Code*.

### 3.1 Ingestie și Preprocesare (`src/data_loader.py`)
1.  **Descărcare Automată:** Scriptul interoghează API-ul Open-Meteo doar dacă datele nu există local, asigurând eficiență.
2.  **Curățare:** Se elimină metadatele inutile și se redenumesc coloanele pentru consistență.
3.  **Interpolare:** Valorile lipsă (dacă există) sunt completate prin interpolare liniară temporală, menținând continuitatea seriilor de timp.
4.  **Normalizare:** Se aplică algoritmul **MinMax Scaler**. Toate valorile sunt scalate în intervalul `[0, 1]`, pas critic pentru convergența optimă a rețelei neuronale.

### 3.2 Strategia de Împărțire a Datelor (`src/split_data.py`)
Pentru a elimina **bias-ul sezonier** și a garanta o evaluare corectă a modelului, am implementat o strategie avansată de segmentare cronologică:

| Set de Date | Perioada / Logica | Descriere |
| :--- | :--- | :--- |
| **Train** | **2020 – 2023** (4 Ani) | Modelul învață tiparele istorice și trendurile multianuale. |
| **Validation** | **2024 (Luni Impare)** | Ian, Mar, Mai, Iul, Sep, Nov. Folosit pentru ajustarea hiperparametrilor. |
| **Test** | **2024 (Luni Pare)** | Feb, Apr, Iun, Aug, Oct, Dec. Folosit pentru evaluarea finală. |

> **Notă:** Această abordare "alternată" asigură că modelul este validat și testat pe **toate cele 4 anotimpuri**, prevenind situația în care modelul ar funcționa bine iarna dar ar eșua vara.

---

## 4. Structura Proiectului

```text
Proiect_ReteleNeuronale_Meteo/
├── data/
│   ├── raw/             # Date brute (CSV descărcat automat)
│   ├── processed/       # Dataset normalizat [0, 1]
│   ├── train/           # Setul de antrenare (2020-2023)
│   ├── validation/      # Setul de validare (2024 Luni Impare)
│   └── test/            # Setul de testare (2024 Luni Pare)
├── src/
│   ├── data_loader.py   # Modul de ingestie, curățare și normalizare
│   └── split_data.py    # Modul de segmentare strategică a datelor
├── main.py              # Punctul de intrare (Orchestrator)
├── requirements.txt     # Dependențe (pandas, requests, sklearn)
└── README.md            # Documentația proiectului