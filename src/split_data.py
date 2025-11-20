# src/split_data.py
import pandas as pd
import os

# Definim căile (Paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'bucharest_normalized.csv')

TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')


def split_dataset():
    """Imparte setul de date normalizat in Train (70%), Validation (15%), Test (15%)."""
    print("Se împart datele în seturi de antrenare/testare...")

    if not os.path.exists(INPUT_FILE):
        print(f"Eroare: Nu gasesc fisierul {INPUT_FILE}. Ruleaza data_loader.py intai!")
        return

    # Citim datele
    df = pd.read_csv(INPUT_FILE, index_col=0)  # index_col=0 pastreaza timestamp-ul ca index

    n = len(df)

    # Calculăm punctele de tăiere (70% și 85%)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Feliem datele (Slicing) - NU facem shuffle pentru că sunt serii temporale!
    train_df = df[0:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Salvăm în folderele corespunzătoare
    # Creăm folderele dacă nu există
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(TRAIN_DIR, 'train.csv'))
    val_df.to_csv(os.path.join(VAL_DIR, 'validation.csv'))
    test_df.to_csv(os.path.join(TEST_DIR, 'test.csv'))

    print(f"Distribuția datelor:")
    print(f"Total rânduri: {n}")
    print(f"Train (70%):  {len(train_df)} rânduri -> salvat în data/train/train.csv")
    print(f"Valid (15%):  {len(val_df)} rânduri -> salvat în data/validation/validation.csv")
    print(f"Test  (15%):  {len(test_df)} rânduri -> salvat în data/test/test.csv")


if __name__ == "__main__":
    split_dataset()