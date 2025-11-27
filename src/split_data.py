# src/split_data.py
import pandas as pd
import os
import sys

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'bucharest_normalized.csv')

TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'data', 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')


def split_dataset() -> None:
    """
    Splits the normalized dataset into Train, Validation, and Test sets.

    Strategy:
    - Train: Years 2020-2023 (Historical context).
    - Val/Test: Year 2024 split by alternating months.
      - Validation: Odd months (Jan, Mar, May...)
      - Test: Even months (Feb, Apr, Jun...)

    This ensures both evaluation sets cover all 4 seasons (Winter/Summer bias avoidance).
    """
    print("[INFO] Initiating data split strategy...")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Processed file not found: {INPUT_FILE}")
        sys.exit(1)

    # Load data and ensure index is datetime
    df = pd.read_csv(INPUT_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # --- SPLITTING LOGIC ---

    # 1. Training Set: Everything before 2024
    train_df = df[df.index.year < 2024].copy()

    # 2. Evaluation Data: Only 2024
    df_2024 = df[df.index.year == 2024].copy()

    # 3. Validation vs Test (Alternating Months)
    # Month % 2 != 0 -> Odd months (1, 3, 5...) -> Validation
    # Month % 2 == 0 -> Even months (2, 4, 6...) -> Test
    val_mask = (df_2024.index.month % 2 != 0)
    test_mask = (df_2024.index.month % 2 == 0)

    val_df = df_2024[val_mask]
    test_df = df_2024[test_mask]

    # --- SAVING ---
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(TRAIN_DIR, 'train.csv'))
    val_df.to_csv(os.path.join(VAL_DIR, 'validation.csv'))
    test_df.to_csv(os.path.join(TEST_DIR, 'test.csv'))

    # --- REPORT ---
    print("\n[SUCCESS] Data distribution report:")
    print(f"{'Set':<15} | {'Period':<25} | {'Samples (Hours)':<10}")
    print("-" * 55)
    print(f"{'Train':<15} | {'2020 - 2023':<25} | {len(train_df):<10}")
    print(f"{'Validation':<15} | {'2024 (Odd Months)':<25} | {len(val_df):<10}")
    print(f"{'Test':<15} | {'2024 (Even Months)':<25} | {len(test_df):<10}")
    print("-" * 55)
    print("[INFO] Strategy confirmed: All seasons covered in evaluation sets.")