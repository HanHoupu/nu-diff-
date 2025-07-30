#!/usr/bin/env python3
"""
检查当前数据分布情况
"""
import torch
import yaml
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.train import NucDiffModule, make_loaders_from_preprocessed
from src.nucdiff.utils.paths import get_preprocessed_paths

def plot_nuclide_chart(train_df, val_df, output_path):
    """Generates and saves a chart of nuclides showing train/val split."""
    print("Generating train/val split nuclide chart...")
    
    train_nuclides = train_df[['z', 'a']].drop_duplicates().copy()
    train_nuclides['n'] = train_nuclides['a'] - train_nuclides['z']
    train_nuclides['split'] = 'Train'
    
    val_nuclides = val_df[['z', 'a']].drop_duplicates().copy()
    val_nuclides['n'] = val_nuclides['a'] - val_nuclides['z']
    val_nuclides['split'] = 'Validation'
    
    all_nuclides = pd.concat([train_nuclides, val_nuclides])
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(data=all_nuclides, x='n', y='z', hue='split', alpha=0.8, s=20, palette={'Train': 'C0', 'Validation': 'C1'})
    
    magic_numbers = [2, 8, 20, 28, 50, 82, 126]
    for m in magic_numbers:
        plt.axvline(x=m, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.axhline(y=m, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.title('Train/Validation Split on Chart of Nuclides', fontsize=16)
    plt.xlabel('Neutron Number (N)', fontsize=12)
    plt.ylabel('Proton Number (Z)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Dataset Split')
    plt.tight_layout()
    plt.savefig(output_path / "0_train_val_split_chart.png")
    plt.close()

def analyze_model_predictions(checkpoint_path: str, output_dir: str):
    """
    Loads a trained model, runs predictions on the validation set,
    and generates diagnostic plots for error analysis.
    """
    print(f"--- Analysis Started for Model: {checkpoint_path} ---")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    run_dir = ckpt_path.parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in run directory: {run_dir}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Analysis results will be saved to: {output_path.resolve()}")

    # --- 1. Data Split Analysis ---
    train_path, val_path = get_preprocessed_paths(cfg, 'L')
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Train/Val data not found at {train_path.parent}. Please run preprocess.py first.")
    
    train_df_original = pd.read_feather(train_path)
    val_df_original = pd.read_feather(val_path)
    
    plot_nuclide_chart(train_df_original, val_df_original, output_path)

    # --- 2. Model Performance Analysis ---
    print("\n--- Starting Model Performance Analysis ---")
    _, val_loader, meta = make_loaders_from_preprocessed(cfg, 'L', debug=False)
    
    print("Loading model from checkpoint...")
    model = NucDiffModule.load_from_checkpoint(
        checkpoint_path, 
        cfg=cfg, 
        task='L', 
        elem2idx=meta['elem2idx'], 
        rec2idx=meta['rec2idx'],
        map_location=torch.device('cpu')
    )
    model.eval()
    
    print("Running predictions on the validation set...")
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in val_loader:
            batch_x, batch_y = batch
            pred = model(batch_x)
            
            mask = torch.isfinite(batch_y['L'])
            if mask.any():
                predictions.extend(pred[mask].cpu().numpy())
                actuals.extend(batch_y['L'][mask].cpu().numpy())

    results_df = val_df_original[val_df_original['energy_keV'].notna()].copy()
    
    if len(results_df) == len(predictions):
        results_df['predicted'] = predictions
        results_df.rename(columns={'energy_keV': 'actual'}, inplace=True)
    else:
        print(f"Warning: Length mismatch! Predictions: {len(predictions)}, Valid non-NaN rows: {len(results_df)}")
        results_df = pd.DataFrame({'predicted': predictions, 'actual': actuals})

    results_df['error'] = results_df['predicted'] - results_df['actual']
    results_df['abs_error'] = results_df['error'].abs()

    print("Generating model performance plots...")

    # Plot 1: Prediction vs. Actual
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=results_df, x='actual', y='predicted', alpha=0.5)
    plt.plot([results_df['actual'].min(), results_df['actual'].max()], [results_df['actual'].min(), results_df['actual'].max()], 'r--', lw=2, label="y=x")
    plt.title('Prediction vs. Actual Values')
    plt.xlabel('Actual Energy (keV)')
    plt.ylabel('Predicted Energy (keV)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path / "1_prediction_vs_actual.png")
    plt.close()

    # Plot 2: Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='error', bins=50, kde=True)
    plt.title('Distribution of Prediction Errors (predicted - actual)')
    plt.xlabel('Error (keV)')
    plt.grid(True)
    plt.savefig(output_path / "2_error_distribution.png")
    plt.close()

    # Plot 3: Absolute Error vs. Actual Value
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='actual', y='abs_error', alpha=0.5)
    plt.title('Absolute Error vs. Actual Value')
    plt.xlabel('Actual Energy (keV)')
    plt.ylabel('Absolute Error (keV)')
    plt.grid(True)
    plt.savefig(output_path / "3_abs_error_vs_actual.png")
    plt.close()

    # Plot 4: Nuclide-level Error Analysis
    if 'z' in results_df.columns and 'a' in results_df.columns:
        nuclide_errors = results_df.groupby(['z', 'a'])['abs_error'].agg(['mean', 'count']).rename(
            columns={'mean': 'mean_abs_error'}
        )
        nuclide_errors.sort_values(by='mean_abs_error', ascending=False, inplace=True)
        
        print("\n--- Top 20 Nuclides with Highest Mean Absolute Error ---")
        print(nuclide_errors.head(20).to_string())

        plt.figure(figsize=(12, 10))
        top_20_errors = nuclide_errors.head(20).sort_values(by='mean_abs_error', ascending=True)
        top_20_errors.reset_index(inplace=True)
        top_20_errors['nuclide'] = top_20_errors.apply(lambda row: f"Z={int(row['z'])}, A={int(row['a'])}", axis=1)
        
        sns.barplot(data=top_20_errors, y='nuclide', x='mean_abs_error', palette='viridis')
        plt.title('Top 20 Nuclides with Highest Mean Absolute Error')
        plt.xlabel('Mean Absolute Error (keV)')
        plt.ylabel('Nuclide (Z, A)')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path / "4_top_20_error_nuclides.png")
        plt.close()

    print(f"\n--- Analysis Summary ---")
    print(f"Mean Absolute Error: {results_df['abs_error'].mean():.4f}")
    if len(results_df['actual']) > 1 and results_df['actual'].std() > 0:
        r2 = 1 - (results_df['error']**2).sum() / ((results_df['actual'] - results_df['actual'].mean())**2).sum()
        print(f"Calculated R² score: {r2:.4f}")
    else:
        print("Could not calculate R² score.")

def main():
    parser = argparse.ArgumentParser(description="Model and Data Analysis Tool")
    parser.add_argument("--ckpt", required=True, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument("--out", default="analysis_results", help="Directory to save the analysis plots.")
    args = parser.parse_args()
    
    analyze_model_predictions(checkpoint_path=args.ckpt, output_dir=args.out)

if __name__ == "__main__":
    main() 