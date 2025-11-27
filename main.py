# main.py
import sys
from src.data_loader import download_data, process_data
from src.split_data import split_dataset

def main():
    print("\n" + "="*50)
    print("   WEATHER AI PROJECT - DATA PIPELINE")
    print("="*50 + "\n")

    # Step 1: Ingestion
    print(">>> STEP 1: Data Ingestion")
    download_data()
    print("")

    # Step 2: Processing
    print(">>> STEP 2: Data Preprocessing (Normalization)")
    process_data()
    print("")

    # Step 3: Splitting
    print(">>> STEP 3: Dataset Splitting")
    split_dataset()

    print("\n" + "="*50)
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        sys.exit(1)