# main.py
from src.data_loader import download_data, process_data
from src.split_data import split_dataset

if __name__ == "__main__":
    print("-- Pornim proiectul --")
    print("\n-- Descarcarea datelor daca nu exista si procesarea acestora --")

    # 1. Descarcare
    download_data()

    # 2. Procesare
    process_data()

    # 3. Impartire date
    split_dataset()

    print("\n-- Procesarea datelor finalizata. --")