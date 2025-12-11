# main.py
import sys
# Update imports based on new folder structure
from src.data_acquisition.data_loader import download_data
from src.data_acquisition.synthetic_generator import generate_synthetic_data
from src.processing.split_data import split_and_normalize_dataset  # <-- Noul import


def main():
    print("\n" + "=" * 50)
    print("   SIA-METEO: FULL PIPELINE START")
    print("=" * 50 + "\n")

    # Step 1: Real Data
    print(">>> STEP 1: Real Data Acquisition")
    download_data()
    print("")

    # Step 2: Synthetic Generation
    print(">>> STEP 2: Synthetic Data Generation (Black Swan Events)")
    generate_synthetic_data()
    print("")

    # Step 3: Processing & Splitting
    print(">>> STEP 3: Splitting & Normalization")
    split_and_normalize_dataset()

    print("\n" + "=" * 50)
    print("   PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}")
        # Print full trace for debugging if needed
        import traceback

        traceback.print_exc()
        sys.exit(1)