#!/usr/bin/env python3
"""
Test new Stratified + Group data splitting method
"""
import sys
sys.path.append('src')

from nucdiff.data.temporal_dataloader import load_temporal_data, stratified_split_by_ZA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_stratified_split():
    """Test the new splitting method"""
    
    print("=" * 50)
    print("Testing Stratified + Group Data Split")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = load_temporal_data(2004, 2010, "L")
    print(f"Total data size: {len(df)}")
    
    # 2. Use new splitting method
    print("\n2. Using new splitting method...")
    train_df, val_df = stratified_split_by_ZA(df, val_frac=0.2)
    
    # 3. Detailed result analysis
    print("\n3. Detailed result analysis...")
    
    # Energy distribution comparison
    train_energy = train_df['energy_keV']
    val_energy = val_df['energy_keV']
    
    print(f"\nEnergy statistics comparison:")
    print(f"Train: mean={train_energy.mean():.3f}, std={train_energy.std():.3f}, var={train_energy.var():.3f}")
    print(f"Val: mean={val_energy.mean():.3f}, std={val_energy.std():.3f}, var={val_energy.var():.3f}")
    print(f"Variance ratio: {val_energy.var() / train_energy.var():.3f}")
    
    # Nucleus overlap check
    train_nuclei = set(train_df[['z', 'a']].apply(tuple, axis=1))
    val_nuclei = set(val_df[['z', 'a']].apply(tuple, axis=1))
    common_nuclei = train_nuclei & val_nuclei
    
    print(f"\nNucleus distribution:")
    print(f"Train nucleus count: {len(train_nuclei)}")
    print(f"Val nucleus count: {len(val_nuclei)}")
    print(f"Overlapping nucleus count: {len(common_nuclei)}")
    
    if len(common_nuclei) > 0:
        print(f"Overlapping nucleus ratio: {len(common_nuclei) / len(train_nuclei | val_nuclei) * 100:.1f}%")
    
    # Energy range distribution
    print(f"\nEnergy range distribution:")
    energy_ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, float('inf'))]
    
    for low, high in energy_ranges:
        if high == float('inf'):
            range_str = f">{low} keV"
            train_count = len(train_df[train_df['energy_keV'] > low])
            val_count = len(val_df[val_df['energy_keV'] > low])
        else:
            range_str = f"{low}-{high} keV"
            train_count = len(train_df[(train_df['energy_keV'] >= low) & (train_df['energy_keV'] < high)])
            val_count = len(val_df[(val_df['energy_keV'] >= low) & (val_df['energy_keV'] < high)])
        
        train_pct = train_count / len(train_df) * 100
        val_pct = val_count / len(val_df) * 100
        
        print(f"  {range_str:>12}: Train {train_count:>6} ({train_pct:>5.1f}%), Val {val_count:>6} ({val_pct:>5.1f}%)")
    
    # 4. Visualization comparison
    print("\n4. Generating visualization comparison...")
    
    plt.figure(figsize=(15, 10))
    
    # Energy distribution histogram
    plt.subplot(2, 3, 1)
    plt.hist(train_energy, bins=50, alpha=0.6, label='Train', color='blue', density=True)
    plt.hist(val_energy, bins=50, alpha=0.6, label='Val', color='red', density=True)
    plt.xlabel('Energy (keV)')
    plt.ylabel('Density')
    plt.title('Energy Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution function
    plt.subplot(2, 3, 2)
    train_sorted = np.sort(train_energy)
    val_sorted = np.sort(val_energy)
    train_cdf = np.arange(1, len(train_sorted) + 1) / len(train_sorted)
    val_cdf = np.arange(1, len(val_sorted) + 1) / len(val_sorted)
    
    plt.plot(train_sorted, train_cdf, label='Train', color='blue')
    plt.plot(val_sorted, val_cdf, label='Val', color='red')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Quantile comparison
    plt.subplot(2, 3, 3)
    percentiles = np.linspace(0, 100, 21)
    train_quantiles = np.percentile(train_energy, percentiles)
    val_quantiles = np.percentile(val_energy, percentiles)
    
    plt.plot(percentiles, train_quantiles, 'o-', label='Train', color='blue')
    plt.plot(percentiles, val_quantiles, 'o-', label='Val', color='red')
    plt.xlabel('Percentile (%)')
    plt.ylabel('Energy (keV)')
    plt.title('Quantile Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Energy range distribution
    plt.subplot(2, 3, 4)
    ranges = ['0-100', '100-500', '500-1000', '1000-2000', '>2000']
    train_counts = []
    val_counts = []
    
    for low, high in energy_ranges:
        if high == float('inf'):
            train_count = len(train_df[train_df['energy_keV'] > low])
            val_count = len(val_df[val_df['energy_keV'] > low])
        else:
            train_count = len(train_df[(train_df['energy_keV'] >= low) & (train_df['energy_keV'] < high)])
            val_count = len(val_df[(val_df['energy_keV'] >= low) & (val_df['energy_keV'] < high)])
        
        train_counts.append(train_count / len(train_df) * 100)
        val_counts.append(val_count / len(val_df) * 100)
    
    x = np.arange(len(ranges))
    width = 0.35
    
    plt.bar(x - width/2, train_counts, width, label='Train', alpha=0.7, color='blue')
    plt.bar(x + width/2, val_counts, width, label='Val', alpha=0.7, color='red')
    plt.xlabel('Energy Range (keV)')
    plt.ylabel('Percentage (%)')
    plt.title('Energy Range Distribution')
    plt.xticks(x, ranges)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistical metrics comparison
    plt.subplot(2, 3, 5)
    metrics = ['mean', 'std', 'min', 'max', 'var']
    train_vals = [train_energy.mean(), train_energy.std(), train_energy.min(), train_energy.max(), train_energy.var()]
    val_vals = [val_energy.mean(), val_energy.std(), val_energy.min(), val_energy.max(), val_energy.var()]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, train_vals, width, label='Train', alpha=0.7, color='blue')
    plt.bar(x + width/2, val_vals, width, label='Val', alpha=0.7, color='red')
    plt.xlabel('Statistical Metrics')
    plt.ylabel('Value')
    plt.title('Statistical Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Nucleus distribution
    plt.subplot(2, 3, 6)
    categories = ['Train Only', 'Val Only', 'Overlap']
    counts = [len(train_nuclei - val_nuclei), len(val_nuclei - train_nuclei), len(common_nuclei)]
    colors = ['blue', 'red', 'orange']
    
    plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Nucleus Distribution')
    
    plt.tight_layout()
    plt.savefig('stratified_split_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Chart saved as 'stratified_split_analysis.png'")

if __name__ == "__main__":
    test_stratified_split() 