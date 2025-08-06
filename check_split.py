# check_split.py
import pandas as pd
from pathlib import Path
import argparse
import sys

def find_latest_files(data_dir: Path, task: str = "SEQ"):
    """Finds the latest train and validation feather files for a given task."""
    train_files = list(data_dir.glob(f"task_{task}_*_train.feather"))
    val_files = list(data_dir.glob(f"task_{task}_*_val.feather"))

    if not train_files:
        print(f"❌ No training files found in '{data_dir}' for task '{task}'.")
        return None, None

    # Sort by creation time to get the most recent ones
    latest_train = sorted(train_files, key=lambda f: f.stat().st_ctime, reverse=True)[0]
    
    # Find the corresponding validation file
    # Example train file name: task_SEQ_years_2005-2010_seed_412_frac_0.80_train.feather
    file_stem = latest_train.name.replace('_train.feather', '')
    latest_val = data_dir / f"{file_stem}_val.feather"

    if not latest_val.exists():
        print(f"⚠️ Warning: Corresponding validation file not found for {latest_train.name}.")
        return latest_train, None
        
    return latest_train, latest_val

def check_for_leakage(train_path: Path, val_path: Path):
    """
    Checks for nuclide-level leakage between a training and a validation set.
    """
    print(f"--- Checking for data leakage ---")
    print(f"  Train file: {train_path.name}")
    print(f"  Val file:   {val_path.name}")

    try:
        train_df = pd.read_feather(train_path)
        val_df = pd.read_feather(val_path)
    except Exception as e:
        print(f"❌ Error reading data files: {e}")
        return

    # Ensure required columns exist
    required_cols = ['Z_int', 'A_int']
    if not all(col in train_df.columns for col in required_cols) or \
       not all(col in val_df.columns for col in required_cols):
        print(f"❌ Required columns {required_cols} not found in one or both files.")
        return

    # Extract unique nuclides from each set
    train_nuclides = set(zip(train_df['Z_int'], train_df['A_int']))
    val_nuclides = set(zip(val_df['Z_int'], val_df['A_int']))

    print(f"\n🔬 Analysis:")
    print(f"  Unique nuclides in training set:   {len(train_nuclides)}")
    print(f"  Unique nuclides in validation set: {len(val_nuclides)}")

    # Find the intersection (the leakage)
    leaked_nuclides = train_nuclides.intersection(val_nuclides)

    print("\n--- Results ---")
    if not leaked_nuclides:
        print("✅ SUCCESS: No nuclide-level data leakage detected between the training and validation sets.")
    else:
        print(f"🚨 FAILURE: Data leakage detected!")
        print(f"  {len(leaked_nuclides)} nuclides were found in BOTH the training and validation sets.")
        print("  Leaked (Z, A) pairs:")
        # Print a sample of leaked nuclides
        for i, nuclide in enumerate(list(leaked_nuclides)[:10]):
            print(f"    - {nuclide}")
        if len(leaked_nuclides) > 10:
            print("    - ... (and more)")
    print("-" * 15)


def main():
    parser = argparse.ArgumentParser(description="Check for data leakage between train and validation sets.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("preprocessed_data"),
        help="Directory containing the preprocessed .feather files."
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"❌ Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)

    train_file, val_file = find_latest_files(args.data_dir)

    if train_file and val_file:
        check_for_leakage(train_file, val_file)
    else:
        print("\nCould not perform check due to missing files.")


if __name__ == "__main__":
    main()
