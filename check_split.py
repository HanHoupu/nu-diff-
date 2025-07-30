import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
import numpy as np

# --- Setup Paths ---
# This allows the script to find the src module
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.utils.paths import get_preprocessed_paths

def visualize_split(config_path: str, output_filename: str):
    """
    Loads preprocessed train/val data based on a config file and generates
    a nuclide chart visualizing the split.
    """
    print(f"--- Analyzing Data Split from Config: {config_path} ---")
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at: {cfg_path}")

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # --- 1. Load Data ---
    train_path, val_path = get_preprocessed_paths(cfg, 'L')
    print(f"Attempting to load train data from: {train_path}")
    print(f"Attempting to load validation data from: {val_path}")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Train/Val data not found at {train_path.parent}. "
            "Please run preprocess.py with the correct config values first."
        )
    
    train_df = pd.read_feather(train_path)
    val_df = pd.read_feather(val_path)
    print(f"Loaded {len(train_df)} train records and {len(val_df)} validation records.")

    # --- 2. Generate Nuclide Chart ---
    print("Generating train/val split nuclide chart...")
    
    train_nuclides = train_df[['z', 'a']].drop_duplicates().copy()
    train_nuclides['n'] = train_nuclides['a'] - train_nuclides['z']
    train_nuclides['split'] = 'Train'
    
    val_nuclides = val_df[['z', 'a']].drop_duplicates().copy()
    val_nuclides['n'] = val_nuclides['a'] - val_nuclides['z']
    val_nuclides['split'] = 'Validation'
    
    all_nuclides = pd.concat([train_nuclides, val_nuclides])
    
    plt.figure(figsize=(18, 11))
    sns.scatterplot(data=all_nuclides, x='n', y='z', hue='split', alpha=0.9, s=25, palette={'Train': '#1f77b4', 'Validation': '#ff7f0e'})
    
    magic_numbers = [2, 8, 20, 28, 50, 82, 126]
    for m in magic_numbers:
        plt.axvline(x=m, color='grey', linestyle='--', linewidth=1.0, alpha=0.7)
        plt.axhline(y=m, color='grey', linestyle='--', linewidth=1.0, alpha=0.7)

    plt.title('Train vs. Validation Split on the Chart of Nuclides', fontsize=18)
    plt.xlabel('Neutron Number (N)', fontsize=14)
    plt.ylabel('Proton Number (Z)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title='Dataset Split', fontsize=12)
    plt.tight_layout()
    
    output_path = Path(output_filename)
    plt.savefig(output_path)
    print(f"\nSuccess! Chart saved to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the train/validation split of preprocessed data.")
    parser.add_argument(
        "--cfg", 
        required=True, 
        help="Path to the configuration file (.yaml) used during preprocessing."
    )
    parser.add_argument(
        "--out",
        default="split_visualization.png",
        help="Output filename for the plot."
    )
    args = parser.parse_args()
    
    visualize_split(config_path=args.cfg, output_filename=args.out)

if __name__ == "__main__":
    main() 