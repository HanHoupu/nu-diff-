# src/nucdiff/data/temporal_dataloader.py

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import re
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple
import json
import math
from .dataloader import iter_l_records, iter_g_records, iter_q_records, ENSDFDataset

def load_temporal_data(start_year: int, end_year: int, task: str = "L") -> pd.DataFrame:
    """
    Load data within the specified time range
    
    Args:
        start_year: start year
        end_year: end year
        task: task type ("L", "G", "Q")
    
    Returns:
        DataFrame with temporal data
    """
    data_dir = Path("clean data")
    all_records = []
    
    for year in range(start_year, end_year + 1):
        file_pattern = f"ENSDF-{year}.{task}"
        file_path = data_dir / file_pattern
        
        if file_path.exists():
            if task == "L":
                records = list(iter_l_records(file_path))
            elif task == "G":
                records = list(iter_g_records(file_path))
            elif task == "Q":
                records = list(iter_q_records(file_path))
            else:
                raise ValueError(f"Unknown task: {task}")
            
            all_records.extend(records)
    
    if not all_records:
        raise ValueError(f"No data found for {task} task in years {start_year}-{end_year}")
    
    df = pd.DataFrame(all_records)
    print(f"Loaded {len(df)} {task} records from {start_year}-{end_year}")
    return df

def stratified_split_by_ZA(df: pd.DataFrame, val_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified + Group split: Ensure balanced energy distribution while preventing nucleus leakage
    
    Strategy:
    1. First stratify by energy levels, divide data into energy bins
    2. Within each energy bin, group by (Z,A) to ensure same nucleus records stay together
    3. Split groups within each energy bin to ensure balanced distribution
    
    Args:
        df: input data
        val_frac: validation set fraction
    
    Returns:
        train_df, val_df
    """
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np
    
    # 1. Create energy stratification labels (10 quantile bins)
    try:
        energy_bins = pd.qcut(df['energy_keV'], q=10, labels=False, duplicates='drop')
        n_bins = len(np.unique(energy_bins))
        print(f"Created {n_bins} energy bins")
    except Exception as e:
        print(f"Energy stratification failed, using simple percentiles: {e}")
        # If qcut fails, use simple percentiles
        energy_percentiles = np.percentile(df['energy_keV'], np.linspace(0, 100, 11))
        energy_bins = pd.cut(df['energy_keV'], bins=energy_percentiles, labels=False, include_lowest=True)
        n_bins = len(np.unique(energy_bins))
    
    # 2. Create nucleus group identifiers
    df_work = df.copy()
    df_work['energy_bin'] = energy_bins
    df_work['ZA_group'] = df_work['z'].astype(str) + '_' + df_work['a'].astype(str)
    
    # 3. Collect all unique nucleus groups first
    all_nucleus_groups = df_work['ZA_group'].unique()
    
    # 4. For each nucleus group, determine which energy bins it appears in (optimized with groupby)
    print("Mapping nuclei to energy bins...")
    nucleus_to_bins = df_work.groupby('ZA_group')['energy_bin'].unique().to_dict()
    
    # 5. Assign each nucleus group entirely to train or val
    np.random.seed(42)  # For reproducibility
    shuffled_nuclei = np.random.permutation(all_nucleus_groups)
    val_nucleus_count = int(len(all_nucleus_groups) * val_frac)
    
    val_nuclei = set(shuffled_nuclei[:val_nucleus_count])
    train_nuclei = set(shuffled_nuclei[val_nucleus_count:])
    
    # 6. Split data based on nucleus assignment
    train_indices = []
    val_indices = []
    
    print("Assigning nuclei to train/val sets...")
    for nucleus in tqdm(all_nucleus_groups, desc="Assigning Groups", ncols=80):
        nucleus_data_indices = df_work.index[df_work['ZA_group'] == nucleus]
        if nucleus in val_nuclei:
            val_indices.extend(nucleus_data_indices)
        else:
            train_indices.extend(nucleus_data_indices)
    
    # 7. Create final train and validation sets
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    
    # 8. Validate results
    print(f"Split results: Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Check nucleus leakage
    train_nuclei_final = set(train_df[['z', 'a']].apply(tuple, axis=1))
    val_nuclei_final = set(val_df[['z', 'a']].apply(tuple, axis=1))
    common_nuclei = train_nuclei_final & val_nuclei_final
    
    if len(common_nuclei) > 0:
        print(f"⚠️  Found {len(common_nuclei)} overlapping nuclei, potential leakage")
    else:
        print("✅ No nucleus leakage")
    
    # Check energy distribution
    train_energy_stats = {
        'mean': train_df['energy_keV'].mean(),
        'std': train_df['energy_keV'].std(),
        'var': train_df['energy_keV'].var()
    }
    
    val_energy_stats = {
        'mean': val_df['energy_keV'].mean(),
        'std': val_df['energy_keV'].std(),
        'var': val_df['energy_keV'].var()
    }
    
    print(f"Train energy: mean={train_energy_stats['mean']:.3f}, std={train_energy_stats['std']:.3f}")
    print(f"Val energy: mean={val_energy_stats['mean']:.3f}, std={val_energy_stats['std']:.3f}")
    print(f"Variance ratio: {val_energy_stats['var'] / train_energy_stats['var']:.3f}")
    
    return train_df, val_df

def build_temporal_dataset(
    base_start_year: int,
    base_end_year: int,
    val_start_year: int,
    val_end_year: int,
    task: str = "L",
    cfg: dict = None
) -> Tuple[ENSDFDataset, ENSDFDataset, dict]:
    """
    Build temporal dataset
    
    Args:
        base_start_year: base training start year
        base_end_year: base training end year
        val_start_year: validation start year
        val_end_year: validation end year
        task: task type
        cfg: configuration dictionary
    
    Returns:
        train_dataset, val_dataset, metadata
    """
    # Load complete year range data
    print(f"加载 {base_start_year}-{base_end_year} 年数据")
    base_df = load_temporal_data(base_start_year, base_end_year, task)
    # Merge data for grouping
    all_df = base_df.copy().reset_index(drop=True)
    # Group by (Z,A), assign isotopes to train or val
    gss = GroupShuffleSplit(test_size=1-cfg.get("train_frac", 0.8), random_state=cfg.get("seed", 42))
    groups = all_df[["z", "a"]].apply(lambda x: f"{x['z']}_{x['a']}", axis=1)
    train_idx, val_idx = next(gss.split(all_df, groups=groups))
    train_df = all_df.iloc[train_idx].reset_index(drop=True)
    val_df = all_df.iloc[val_idx].reset_index(drop=True)
    
    # Build features
    numeric_cols = []
    if task == "L":
        numeric_cols = ['energy_keV']
    elif task == "G":
        numeric_cols = ['e_gamma_keV']
    elif task == "Q":
        numeric_cols = ['q_value_keV']
    
    # Create element and record type mappings
    elem2idx = {elem: idx for idx, elem in enumerate(train_df['element'].unique())}
    rec2idx = {task: 0}  # Single task has only one record type
    
    # Add necessary index fields
    train_df['element_idx'] = train_df['element'].map(elem2idx)
    train_df['record_type_idx'] = 0  # Single task has only one record type
    val_df['element_idx'] = val_df['element'].map(elem2idx)
    val_df['record_type_idx'] = 0  # Single task has only one record type
    
    # Add target fields (single task training)
    if task == "L":
        # Only use valid energy data as labels
        train_df['target_L'] = train_df['energy_keV'].where(train_df['energy_keV'].notna(), np.nan)
        train_df['target_G'] = np.nan
        train_df['target_Q'] = np.nan
        val_df['target_L'] = val_df['energy_keV'].where(val_df['energy_keV'].notna(), np.nan)
        val_df['target_G'] = np.nan
        val_df['target_Q'] = np.nan
    elif task == "G":
        train_df['target_L'] = np.nan
        train_df['target_G'] = train_df['e_gamma_keV']
        train_df['target_Q'] = np.nan
        val_df['target_L'] = np.nan
        val_df['target_G'] = val_df['e_gamma_keV']
        val_df['target_Q'] = np.nan
    elif task == "Q":
        train_df['target_L'] = np.nan
        train_df['target_G'] = np.nan
        train_df['target_Q'] = train_df['q_value_keV']
        val_df['target_L'] = np.nan
        val_df['target_G'] = np.nan
        val_df['target_Q'] = val_df['q_value_keV']
    
    # Create dataset (let ENSDFDataset calculate statistics)
    train_dataset = ENSDFDataset(train_df, numeric_cols, elem2idx, rec2idx, is_train=True)
    
    # Get training set statistics
    stats = {
        'L_mean': train_dataset.L_mean,
        'L_std': train_dataset.L_std,
        'G_mean': train_dataset.G_mean,
        'G_std': train_dataset.G_std,
        'Q_mean': train_dataset.Q_mean,
        'Q_std': train_dataset.Q_std,
    }
    
    val_dataset = ENSDFDataset(val_df, numeric_cols, elem2idx, rec2idx, is_train=False, stats=stats)
    
    metadata = {
        'elem2idx': elem2idx,
        'rec2idx': rec2idx,
        'numeric_dim': len(numeric_cols),
        'stats': stats
    }
    
    return train_dataset, val_dataset, metadata

def build_multi_task_temporal_dataset(
    base_start_year: int,
    base_end_year: int,
    val_start_year: int,
    val_end_year: int,
    cfg: dict = None
) -> Tuple[ENSDFDataset, ENSDFDataset, dict]:
    """
    Build multi-task temporal dataset (for LoRA fine-tuning)
    
    Args:
        base_start_year: base training start year
        base_end_year: base training end year
        val_start_year: validation start year
        val_end_year: validation end year
        cfg: configuration dictionary
    
    Returns:
        train_dataset, val_dataset, metadata
    """
    # Load data for all tasks
    tasks = ["L", "G", "Q"]
    all_dfs = {}
    
    for task in tasks:
        df = load_temporal_data(base_start_year, base_end_year, task)
        all_dfs[task] = df
    
    # Merge all task data
    # Adjust based on actual data structure
    # Use a simple merge strategy for now
    combined_df = pd.concat(all_dfs.values(), ignore_index=True)
    
    # Create training/validation masks
    train_mask = combined_df['dataset_year'].between(base_start_year, base_end_year)
    val_mask = combined_df['dataset_year'].between(val_start_year, val_end_year)
    
    train_df = combined_df[train_mask].copy()
    val_df = combined_df[val_mask].copy()
    
    # Build features
    numeric_cols = ['energy_keV', 'e_gamma_keV', 'q_value_keV']
    
    # Create element and record type mappings
    elem2idx = {elem: idx for idx, elem in enumerate(train_df['element'].unique())}
    rec2idx = {task: idx for idx, task in enumerate(tasks)}
    
    # Calculate statistics
    stats = {}
    for col in numeric_cols:
        valid_data = train_df[col].dropna()
        if len(valid_data) > 0:
            stats[f'{col}_mean'] = valid_data.mean()
            stats[f'{col}_std'] = valid_data.std()
        else:
            stats[f'{col}_mean'] = 0.0
            stats[f'{col}_std'] = 1.0
    
    # Create dataset
    train_dataset = ENSDFDataset(train_df, numeric_cols, elem2idx, rec2idx, is_train=True, stats=stats)
    val_dataset = ENSDFDataset(val_df, numeric_cols, elem2idx, rec2idx, is_train=False, stats=stats)
    
    metadata = {
        'elem2idx': elem2idx,
        'rec2idx': rec2idx,
        'numeric_dim': len(numeric_cols),
        'stats': stats
    }
    
    return train_dataset, val_dataset, metadata 