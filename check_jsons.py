# check_jsons.py
"""
A comprehensive script to check for data leakage in preprocessed JSON files.
It verifies:
1. Vocabulary Safety: Ensures the vocabulary is built only from the training set.
2. Index Map Safety: Ensures the year_index_map.json (for Route-2) does not contain any nuclides from the validation set.
"""
import pandas as pd
from pathlib import Path
import argparse
import sys
import json
from tqdm import tqdm

def find_latest_files(data_dir: Path):
    """Finds the latest validation feather file, tokenizer, and year index map."""
    val_files = list(data_dir.glob("task_SEQ_*_val.feather"))
    vocab_files = list(data_dir.glob("tokenizer_vocab.json"))
    map_files = list(data_dir.glob("year_index_map.json"))

    if not val_files or not vocab_files or not map_files:
        print("❌ Error: Missing one or more required files (validation set, vocab, or index map).")
        return None, None, None
        
    latest_val = sorted(val_files, key=lambda f: f.stat().st_ctime, reverse=True)[0]
    latest_vocab = sorted(vocab_files, key=lambda f: f.stat().st_ctime, reverse=True)[0]
    latest_map = sorted(map_files, key=lambda f: f.stat().st_ctime, reverse=True)[0]
    
    return latest_val, latest_vocab, latest_map

def test_vocabulary_safety(val_df: pd.DataFrame, vocab: dict):
    """Test 1: Check for vocabulary leakage."""
    print("\n--- Test 1: Vocabulary Safety Check ---")
    vocab_set = set(vocab.keys())
    
    total_tokens = 0
    unknown_tokens = 0

    for sequence in tqdm(val_df['token_sequence'], desc="1/2: Checking Vocab"):
        tokens = sequence.split(';')
        for token in tokens:
            token = token.strip()
            if not token: continue
            total_tokens += 1
            if token not in vocab_set:
                unknown_tokens += 1
    
    unknown_percentage = (unknown_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"  > Validation tokens not in vocab: {unknown_tokens:,} / {total_tokens:,} ({unknown_percentage:.2f}%)")

    if unknown_percentage == 0.0:
        print("  🚨 RESULT: CRITICAL FAILURE. 0% unknown tokens strongly indicates leakage.")
    else:
        print("  ✅ RESULT: SUCCESS. A healthy percentage of unknown tokens found.")

def test_index_map_safety(val_df: pd.DataFrame, index_map: dict):
    """Test 2: Check for leakage in the year_index_map."""
    print("\n--- Test 2: Year Index Map Safety Check ---")
    
    map_nuclides = set()
    map_data = index_map.get("map", {})
    for key in map_data.keys():
        try:
            z_str, a_str = key.split("_")
            map_nuclides.add((int(z_str), int(a_str)))
        except ValueError:
            continue # Ignore malformed keys if any

    val_nuclides = set(zip(val_df['Z_int'], val_df['A_int']))

    print(f"  > Nuclides in year_index_map (from training set): {len(map_nuclides)}")
    print(f"  > Nuclides in validation set: {len(val_nuclides)}")

    leaked_nuclides = val_nuclides.intersection(map_nuclides)

    if not leaked_nuclides:
        print("  ✅ RESULT: SUCCESS. No validation nuclides found in the training-only index map.")
    else:
        print(f"  🚨 RESULT: CRITICAL FAILURE. Found {len(leaked_nuclides)} validation nuclides in the index map.")
        print(f"     Sample leaked nuclide: {list(leaked_nuclides)[0]}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive check for data leakage in JSON artifacts.")
    parser.add_argument("--data_dir", type=Path, default=Path("preprocessed_data"))
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"❌ Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)

    val_file, vocab_file, map_file = find_latest_files(args.data_dir)

    if not all([val_file, vocab_file, map_file]):
        sys.exit(1)
        
    print("--- Loading Artifacts for Verification ---")
    print(f"  - Validation Set: {val_file.name}")
    print(f"  - Vocabulary:     {vocab_file.name}")
    print(f"  - Index Map:      {map_file.name}")

    try:
        val_df = pd.read_feather(val_file)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        with open(map_file, 'r', encoding='utf-8') as f:
            index_map = json.load(f)
    except Exception as e:
        print(f"❌ Error during file loading: {e}")
        sys.exit(1)

    # Run the checks
    test_vocabulary_safety(val_df, vocab)
    test_index_map_safety(val_df, index_map)
    
    print("\n--- Overall Check Complete ---")

if __name__ == "__main__":
    main()
