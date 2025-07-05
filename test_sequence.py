#!/usr/bin/env python3
"""
Test script for sequence-based data loader
"""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import yaml
from src.nucdiff.data.dataloader import build_dataset

def test_sequence_dataloader():
    print("Testing sequence-based data loader...")
    
    # Load config
    with open('configs/test.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Build dataset
    train_ds, val_ds, (elem2idx, rec2idx, numeric_dim) = build_dataset(2004, cfg)
    
    print(f"Dataset created: {len(train_ds)} train, {len(val_ds)} val samples")
    print(f"Element mapping: {len(elem2idx)} elements")
    print(f"Record type mapping: {len(rec2idx)} record types")
    print(f"Numeric dimension: {numeric_dim}")
    
    # Test a sample
    sample = train_ds[0]
    print(f"\nSample structure:")
    print(f"Input keys: {list(sample[0].keys())}")
    print(f"Input shapes:")
    for key, value in sample[0].items():
        print(f"  {key}: {value.shape}")
    print(f"Target keys: {list(sample[1].keys())}")
    print(f"Target values:")
    for key, value in sample[1].items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ Sequence data loader test passed!")

if __name__ == "__main__":
    test_sequence_dataloader() 