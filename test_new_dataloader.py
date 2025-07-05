#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from nucdiff.data.dataloader import get_loaders

def test_dataloader():
    try:
        print("Testing new dataloader with 2004 data...")
        train_loader, val_loader, maps = get_loaders(2004, {
            'batch_size': 32, 
            'train_frac': 0.8
        })
        
        print(f"✓ Success! Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Test one batch
        for batch_idx, (features, targets) in enumerate(train_loader):
            print(f"✓ Batch {batch_idx}: features={list(features.keys())}, targets={list(targets.keys())}")
            print(f"  Numeric shape: {features['numeric'].shape}")
            print(f"  Element shape: {features['element'].shape}")
            print(f"  Record type shape: {features['record_type'].shape}")
            print(f"  Feature type IDs shape: {features['feature_type_ids'].shape}")
            print(f"  L target shape: {targets['L'].shape}")
            print(f"  G target shape: {targets['G'].shape}")
            print(f"  Q target shape: {targets['Q'].shape}")
            break
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataloader()
    sys.exit(0 if success else 1) 