#!/usr/bin/env python3
import sys
import pathlib
sys.path.append('src')

print("=== Testing 2004 Data Loading ===")

try:
    from nucdiff.data.dataloader import build_dataset
    import yaml
    
    # Load config
    with open('configs/default_clean.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    print("✓ Config loaded")
    
    # Build dataset
    train_ds, val_ds, (elem2idx, rec2idx, numeric_dim) = build_dataset(2004, cfg)
    
    print(f"✓ Dataset created successfully")
    print(f"  - Train samples: {len(train_ds)}")
    print(f"  - Val samples: {len(val_ds)}")
    print(f"  - Numeric dimension: {numeric_dim}")
    print(f"  - Element mapping size: {len(elem2idx)}")
    print(f"  - Record mapping size: {len(rec2idx)}")
    
    # Test a sample
    if len(train_ds) > 0:
        sample_x, sample_y = train_ds[0]
        print(f"  - Sample x keys: {list(sample_x.keys())}")
        print(f"  - Sample y keys: {list(sample_y.keys())}")
        print(f"  - Sample numeric shape: {sample_x['numeric'].shape}")
        print(f"  - Sample targets: L={sample_y['L']:.4f}, G={sample_y['G']:.4f}, Q={sample_y['Q']:.4f}")
    
    print("\n✓ Data loading test passed!")
    print("Ready to start training for 2004...")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc() 