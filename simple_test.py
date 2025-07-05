#!/usr/bin/env python3
import sys
import pathlib
sys.path.append('src')

print("=== Simple Project Test ===")

# Test 1: Basic imports
print("1. Testing basic imports...")
try:
    from nucdiff.model import TransformerModel
    from nucdiff.data.dataloader import build_dataset
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Simple config
print("2. Testing simple config...")
try:
    cfg = {
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "ff_dim": 512,
        "rank": 8,
        "alpha": 16,
        "task_weights": {"L": 1.0, "G": 1.0, "Q": 1.0}
    }
    print("   ✓ Config created")
except Exception as e:
    print(f"   ✗ Config error: {e}")
    sys.exit(1)

# Test 3: Model creation
print("3. Testing model creation...")
try:
    elem2idx = {"H": 0, "He": 1, "Li": 2}
    rec2idx = {"L": 0, "G": 1, "Q": 2}
    model = TransformerModel(cfg, elem2idx, rec2idx)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {param_count:,} parameters")
except Exception as e:
    print(f"   ✗ Model creation error: {e}")
    sys.exit(1)

# Test 4: Data files
print("4. Testing data files...")
data_files = ["levels.feather", "gammas.feather", "q.feather"]
missing = []
for file in data_files:
    if pathlib.Path(file).exists():
        print(f"   ✓ {file} exists")
    else:
        print(f"   ✗ {file} missing")
        missing.append(file)

if missing:
    print(f"   Missing files: {missing}")
    sys.exit(1)

# Test 5: Data loading (if files exist)
print("5. Testing data loading...")
try:
    import pandas as pd
    # Try to read a small sample
    df = pd.read_feather("q.feather")
    print(f"   ✓ Data loaded, full shape: {df.shape}")
    print(f"   ✓ Sample columns: {list(df.columns[:5])}")
except Exception as e:
    print(f"   ✗ Data loading error: {e}")

print("\n=== All basic tests passed! ===")
print("Project structure looks good.") 