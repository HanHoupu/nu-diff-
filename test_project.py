#!/usr/bin/env python3
"""
Simple project test script to verify all components work correctly
"""
import sys
import pathlib
import yaml
import torch
import pandas as pd

# Add src to path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from nucdiff.model import TransformerModel
        from nucdiff.data.dataloader import build_dataset, get_loaders
        from nucdiff.utils.evaluate import evaluate_mae_multi
        from nucdiff.utils.seed import set_seed
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test if config file can be loaded"""
    print("Testing config...")
    try:
        with open("configs/default_clean.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print("✓ Config loaded successfully")
        print(f"  - d_model: {cfg.get('d_model')}")
        print(f"  - batch_size: {cfg.get('batch_size')}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_data_files():
    """Test if data files exist"""
    print("Testing data files...")
    required_files = ["levels.feather", "gammas.feather", "q.feather"]
    missing_files = []
    
    for file in required_files:
        if pathlib.Path(file).exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    return True

def test_model_creation():
    """Test if model can be created"""
    print("Testing model creation...")
    try:
        with open("configs/default_clean.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Create dummy mappings
        elem2idx = {"H": 0, "He": 1, "Li": 2}
        rec2idx = {"L": 0, "G": 1, "Q": 2}
        
        model = TransformerModel(cfg, elem2idx, rec2idx)
        print("✓ Model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_dataloader():
    """Test if dataloader can be created"""
    print("Testing dataloader...")
    try:
        with open("configs/default_clean.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Try to build dataset for year 2004
        train_ds, val_ds, (elem2idx, rec2idx, numeric_dim) = build_dataset(2004, cfg)
        print("✓ Dataset created successfully")
        print(f"  - Train samples: {len(train_ds)}")
        print(f"  - Val samples: {len(val_ds)}")
        print(f"  - Numeric dim: {numeric_dim}")
        
        # Test a sample
        sample_x, sample_y = train_ds[0]
        print(f"  - Sample x keys: {list(sample_x.keys())}")
        print(f"  - Sample y keys: {list(sample_y.keys())}")
        return True
    except Exception as e:
        print(f"✗ Dataloader error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Project Health Check ===\n")
    
    tests = [
        test_imports,
        test_config,
        test_data_files,
        test_model_creation,
        test_dataloader
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"=== Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Project looks healthy.")
    else:
        print("✗ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main() 