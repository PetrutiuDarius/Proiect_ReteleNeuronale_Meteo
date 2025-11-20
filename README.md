# Sistem Inteligent pentru Prognoza Meteorologică

**Student:** Petruțiu Darius-Simion  
**Grupa:** 632AB  
**Proiect:** Rețele Neuronale

---

## 1. Pregătirea Datelor

În această etapă, am realizat infrastructura de date necesară antrenării Rețelei Neuronale (RN), conform arhitecturii sistemului propus în specificațiile inițiale. Obiectivul a fost trecerea de la date brute la seturi de date structurate, normalizate și pregătite pentru învățare supervizată.

### 1.1 Sursa Datelor
Am utilizat date istorice reale pentru **București**, extrase prin API-ul **Open-Meteo (Historical Weather API)**.
- **Perioada:** 01.01.2020 – 01.01.2024.
- **Rezoluție:** Orară (Hourly).
- **Parametri:** Temperatură (2m), Umiditate Relativă, Presiune Atmosferică, Viteza Vântului (10m).

### 1.2 Procesul de Transformare (ETL)

Fluxul de date implementat în scripturile Python (`src/data_loader.py`, `src/split_data.py`) realizează următorii pași:

1.  **Ingestie:** Descărcarea automată a datelor brute în format CSV (`data/raw/`).
2.  **Curățare:**
    - Redenumirea coloanelor pentru consistență.
    - Interpolarea liniară a valorilor lipsă (pentru a menține continuitatea seriilor temporale).
    - Conversia timestamp-urilor la obiecte `datetime`.
3.  **Normalizare:** - Aplicarea algoritmului **MinMax Scaler**.
    - Toate valorile (Temp, Presiune, Vânt, Umiditate) au fost scalate în intervalul `[0, 1]` pentru a asigura convergența optimă a rețelei neuronale.
4.  **Segregarea Datelor:**
    - Datele au fost împărțite cronologic (fără amestecare aleatorie) pentru a respecta natura temporală a problemei.

| Set de Date | Procent | Rol | Fișier rezultat |
| :--- | :--- | :--- | :--- |
| **Train** | 70% | Antrenarea rețelei (ajustarea ponderilor) | `data/train/train.csv` |
| **Validation** | 15% | Monitorizarea performanței și prevenirea overfitting-ului | `data/validation/validation.csv` |
| **Test** | 15% | Evaluarea finală pe date "nevăzute" | `data/test/test.csv` |

---

## 2. Structura Proiectului

```text
Proiect_ReteleNeuronale_Meteo/
├── data/
│   ├── raw/             # Datele brute descărcate de la Open-Meteo
│   ├── processed/       # Dataset-ul complet, curățat și normalizat
│   ├── train/           # Datele pentru antrenare (2020 - 2022)
│   ├── validation/      # Datele pentru validare (2023 partial)
│   └── test/            # Datele pentru testare finală (2023-2024)
├── src/
│   ├── data_loader.py   # Script pentru descărcare și normalizare
│   └── split_data.py    # Script pentru împărțirea dataset-ului
├── main.py              # Punctul de intrare în aplicație
└── requirements.txt     # Dependențele proiectului (pandas, sklearn, requests)