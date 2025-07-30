#!/usr/bin/env python3
"""
Test integrated dataset building functionality
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from pathlib import Path
from nucdiff.data.dataloader import build_unified_dataset, load_records_for_task

def test_unified_dataloader():
    """Test integrated dataset building functionality"""
    print("=== Testing Integrated Dataset Building Functionality ===")
    
    # Mock configuration
    cfg = {
        "train_frac": 0.8,
        "seed": 42,
        "batch_size": 256
    }
    
    try:
        # 1. Test data loading functionality
        print("\n1. Testing data loading functionality...")
        df = load_records_for_task(2004, 2004, "L", debug=True)
        print(f"✅ Data loading successful: {len(df)} records")
        
        # 2. Test single year dataset building
        print("\n2. Testing single year dataset building...")
        train_ds, val_ds, metadata = build_unified_dataset(
            start_year=2004,
            end_year=2004,
            task="L",
            cfg=cfg,
            debug=True
        )
        print(f"✅ Single year dataset building successful!")
        print(f"  - Training set: {len(train_ds)} samples")
        print(f"  - Validation set: {len(val_ds)} samples")
        print(f"  - Element mapping: {len(metadata['elem2idx'])} elements")
        print(f"  - Numeric feature dimension: {metadata['numeric_dim']}")
        
        # 3. Test multi-year dataset building
        print("\n3. Testing multi-year dataset building...")
        train_ds, val_ds, metadata = build_unified_dataset(
            start_year=2004,
            end_year=2006,
            task="L",
            cfg=cfg,
            debug=True
        )
        print(f"✅ Multi-year dataset building successful!")
        print(f"  - Training set: {len(train_ds)} samples")
        print(f"  - Validation set: {len(val_ds)} samples")
        
        # 4. Test different task types
        print("\n4. Testing different task types...")
        for task in ["L", "G", "Q"]:
            try:
                train_ds, val_ds, metadata = build_unified_dataset(
                    start_year=2004,
                    end_year=2004,
                    task=task,
                    cfg=cfg,
                    debug=False
                )
                print(f"✅ {task} task dataset building successful: Train{len(train_ds)} | Val{len(val_ds)}")
            except Exception as e:
                print(f"❌ {task} task dataset building failed: {e}")
        
        # 5. Test data samples
        print("\n5. Testing data samples...")
        sample_x, sample_y = train_ds[0]
        print(f"  - Numeric feature shape: {sample_x['numeric'].shape}")
        print(f"  - Element index: {sample_x['element']}")
        print(f"  - Record type index: {sample_x['record_type']}")
        print(f"  - L label: {sample_y['L']}")
        print(f"  - G label: {sample_y['G']}")
        print(f"  - Q label: {sample_y['Q']}")
        
        # 6. Test DataLoader
        print("\n6. Testing DataLoader...")
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        batch_x, batch_y = next(iter(train_loader))
        print(f"✅ DataLoader working normally!")
        print(f"  - Batch size: {batch_y['L'].shape}")
        print(f"  - Valid L labels count: {(~torch.isnan(batch_y['L'])).sum()}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_mode():
    """Test debug mode"""
    print("\n=== Testing Debug Mode ===")
    
    cfg = {"train_frac": 0.8, "seed": 42}
    
    try:
        # Enable debug mode
        train_ds, val_ds, metadata = build_unified_dataset(
            start_year=2004,
            end_year=2004,
            task="L",
            cfg=cfg,
            debug=True
        )
        print("✅ Debug mode test passed!")
        return True
    except Exception as e:
        print(f"❌ Debug mode test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting to test integrated dataset building functionality...")
    
    # Run tests
    success1 = test_unified_dataloader()
    success2 = test_debug_mode()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Dataset building functionality integration successful!")
    else:
        print("\n❌ Some tests failed, need to check issues.") 