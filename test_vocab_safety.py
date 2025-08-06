# test_vocab_safety.py
import pandas as pd
from pathlib import Path
import argparse
import sys
import json
from tqdm import tqdm

# Add project root to sys.path to allow for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.data.tokenizer import LevelTokenizer

def find_latest_files(data_dir: Path):
    """Finds the latest validation feather file and tokenizer vocab."""
    val_files = list(data_dir.glob("task_SEQ_*_val.feather"))
    vocab_files = list(data_dir.glob("tokenizer_vocab.json"))

    if not val_files:
        return None, None
    if not vocab_files:
        return None, None
        
    latest_val = sorted(val_files, key=lambda f: f.stat().st_ctime, reverse=True)[0]
    latest_vocab = sorted(vocab_files, key=lambda f: f.stat().st_ctime, reverse=True)[0]
    
    return latest_val, latest_vocab

def test_vocabulary_safety(val_path: Path, vocab_path: Path):
    """
    Checks for vocabulary leakage by ensuring some validation tokens are unknown 
    to the training-only vocabulary.
    """
    print("--- Testing Vocabulary Safety ---")
    print(f"  Validation Set: {val_path.name}")
    print(f"  Vocabulary:     {vocab_path.name}")

    try:
        val_df = pd.read_feather(val_path)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # We don't need a full tokenizer, just the vocab set for fast lookups
        vocab_set = set(vocab.keys())
        
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    total_tokens = 0
    unknown_tokens = 0
    unknown_token_examples = set()

    print("\n🔬 Analyzing validation set tokens...")
    # Use tqdm for a progress bar as this can be slow
    for sequence in tqdm(val_df['token_sequence'], desc="Processing sequences"):
        tokens = sequence.split(';')
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            total_tokens += 1
            if token not in vocab_set:
                unknown_tokens += 1
                if len(unknown_token_examples) < 20: # Collect some examples
                    unknown_token_examples.add(token)

    print("\n--- Results ---")
    if total_tokens == 0:
        print("⚠️ Warning: No tokens found in the validation set.")
        return

    unknown_percentage = (unknown_tokens / total_tokens) * 100

    print(f"  Total tokens in validation set: {total_tokens:,}")
    print(f"  Tokens NOT found in vocabulary (unknown): {unknown_tokens:,}")
    print(f"  Percentage of unknown tokens: {unknown_percentage:.4f}%")

    if unknown_percentage == 0.0:
        print("\n🚨 CRITICAL FAILURE: 0% unknown tokens. This strongly indicates vocabulary leakage.")
        print("   This means the vocabulary was likely built using the validation or full dataset.")
        print("   Model performance metrics are likely inflated and unreliable.")
    elif unknown_percentage < 0.01: # A very small threshold
        print("\n⚠️ WARNING: Very low percentage of unknown tokens. Leakage is possible but not certain.")
        print("   This could happen with very large, comprehensive training sets. Manual review recommended.")
    else:
        print("\n✅ SUCCESS: A healthy percentage of unknown tokens was found.")
        print("   This suggests the vocabulary was correctly built using only the training set.")

    if unknown_token_examples:
        print("\n🔍 Sample of tokens in validation set but NOT in vocabulary:")
        for i, example in enumerate(list(unknown_token_examples)):
            print(f"    - '{example}'")
    print("-" * 15)

def main():
    parser = argparse.ArgumentParser(description="Check for vocabulary leakage from the validation set.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("preprocessed_data"),
        help="Directory containing the preprocessed .feather and .json files."
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"❌ Error: Data directory '{args.data_dir}' not found.")
        sys.exit(1)

    val_file, vocab_file = find_latest_files(args.data_dir)

    if val_file and vocab_file:
        test_vocabulary_safety(val_file, vocab_file)
    else:
        print("\nCould not perform check due to missing files (validation set or vocab).")

if __name__ == "__main__":
    main()
