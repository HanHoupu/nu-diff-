import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import numpy as np
import os
import json

from nucdiff.data.dataloader import HybridSeqENSDFDataset, PadCollateHybrid
from nucdiff.data.tokenizer import LevelTokenizer
from nucdiff.train import NucDiffSeq2SeqModule
from nucdiff.utils.paths import PREPROCESSED_DATA_DIR, PROJECT_ROOT, get_preprocessed_paths

def generate_sequence(model, tokenizer, encoder_features, delta_e_mean, delta_e_std, max_len=256):
    """
    Generates a sequence using the trained model with greedy decoding.
    Handles normalization and denormalization of delta_E values.
    """
    model.eval()
    device = next(model.parameters()).device
    
    encoder_features = torch.tensor(encoder_features, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Start with [SOS] token and a delta_E of 0
    token_sequence = [tokenizer.sos_id]
    # The initial dE value is 0, which needs to be normalized before being fed to the model
    delta_e_input_normalized = [(0.0 - delta_e_mean) / (delta_e_std + 1e-6)]
    delta_e_output_keV = [0.0] # This will store the actual keV values

    with torch.no_grad():
        for _ in range(max_len):
            input_tokens = torch.tensor([token_sequence], dtype=torch.long, device=device)
            # Use the sequence of *normalized* dE values as input
            input_delta_e_normalized = torch.tensor([delta_e_input_normalized], dtype=torch.float32, device=device)
            
            batch = {
                'encoder_features': encoder_features,
                'decoder_input_tokens': input_tokens,
                'decoder_input_delta_e': input_delta_e_normalized, # This key is what the model's forward expects
                'attention_mask': torch.ones_like(input_tokens, dtype=torch.bool)
            }
            
            class_logits, regr_output_normalized = model(batch)
            
            next_token_id = class_logits.argmax(dim=-1)[:, -1].item()
            
            if next_token_id == tokenizer.eos_id:
                break
            
            # The model outputs a normalized dE value, we need to denormalize it
            next_delta_e_normalized = regr_output_normalized[:, -1].item()
            next_delta_e_keV = (next_delta_e_normalized * delta_e_std) + delta_e_mean

            token_sequence.append(next_token_id)
            delta_e_input_normalized.append(next_delta_e_normalized)
            delta_e_output_keV.append(next_delta_e_keV)
            
    return token_sequence, delta_e_output_keV


def beam_search_generate(model, tokenizer, encoder_features, beam_width=5, max_len=256):
    """
    Generates a sequence using the trained model with beam search.
    """
    # This is a complex function. For now, we will focus on fixing greedy search
    # and will implement a proper beam search for the hybrid model later if needed.
    # For this fix, we will just have it call greedy search as a placeholder.
    token_seq, delta_e_seq = generate_sequence(model, tokenizer, encoder_features, max_len)
    return token_seq, delta_e_seq


def reconstruct_and_print(tokenizer, tokens, delta_es, true_row):
    """Helper to print truth vs predicted."""
    # Reconstruct predicted sequence
    pred_energy = 0
    pred_parts = []
    # We skip the initial SOS token's dE, so loop starts from index 1
    for i in range(1, len(tokens)):
        pred_energy += delta_es[i]
        token_str = tokenizer.id_to_token.get(tokens[i], "[UNK]")
        pred_parts.append(f"E={pred_energy:.3f},{token_str.split(',', 1)[1]}")
    predicted_sequence = ";".join(pred_parts)

    # Reconstruct true sequence
    true_delta_e = json.loads(true_row['delta_e_sequence'])
    true_tokens = true_row['token_sequence'].split(';')
    true_energy = 0
    true_parts = []
    for i, token in enumerate(true_tokens):
        true_energy += true_delta_e[i]
        true_parts.append(f"E={true_energy:.3f},{token.split(',', 1)[1]}")
    true_sequence = ";".join(true_parts)

    print(f"\n[TRUTH]   : {true_sequence}")
    print(f"[PREDICTED]: {predicted_sequence}")

def main():
    parser = argparse.ArgumentParser(description="Generate sequences from a trained Seq2Seq model.")
    parser.add_argument("--cfg", type=Path, default=PROJECT_ROOT / "configs/temporal.yaml", help="Path to the model config file.")
    parser.add_argument("--checkpoint_path", type=str, 
                        default=None, 
                        help="Path to the trained model checkpoint. If None, finds the latest one.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to predict from the validation set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu).")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search decoding.")
    args = parser.parse_args()

    # --- Load Config, Tokenizer, and Scaler ---
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Determine the preprocessed data directory from the config
    path_cfg = {
        "seed": cfg["seed"], "val_frac": cfg["val_frac"],
        "base_start_year": cfg["base_start_year"], "base_end_year": cfg["base_end_year"]
    }
    _, val_path = get_preprocessed_paths(path_cfg, 'SEQ')
    preprocessed_dir = val_path.parent

    tokenizer = LevelTokenizer.load(preprocessed_dir / "tokenizer_vocab.json")
    print(f"Tokenizer loaded with vocabulary size: {tokenizer.vocab_size}")

    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    delta_e_scaler_info = metadata.get('delta_e_scaler', {'mean': 0.0, 'std': 1.0})
    delta_e_mean = delta_e_scaler_info['mean']
    delta_e_std = delta_e_scaler_info['std']
    print(f"Loaded delta_E scaler: mean={delta_e_mean:.4f}, std={delta_e_std:.4f}")

    # --- Load Model ---
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        print("No checkpoint path provided. Finding the latest best model...")
        run_name = f"temporal_SEQ_{cfg['base_start_year']}-{cfg['base_end_year']}" # This needs to match the run name format
        run_dir = PROJECT_ROOT / "runs"
        # Find the latest directory for this run configuration
        
        # A more robust way to find the latest run directory is needed.
        # For now, this is a placeholder. A better approach would be to parse run names.
        # Let's assume the user provides the path for now.
        # The logic below is simplified and may not work if multiple runs have similar names.
        
        all_run_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not all_run_dirs:
            raise FileNotFoundError(f"No run directories found in {run_dir}")
        latest_run = sorted(all_run_dirs, key=os.path.getmtime, reverse=True)[0]

        checkpoint_dir = latest_run / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("best-*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
        checkpoint_path = checkpoint_files[0]

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = NucDiffSeq2SeqModule.load_from_checkpoint(
        checkpoint_path, 
        cfg=cfg, 
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id,
        delta_e_mean=delta_e_mean,
        delta_e_std=delta_e_std,
        # We need to provide encoder_feature_dim, let's calculate it
        encoder_feature_dim=10 # Z, A, asym, is_odd_Z/N, delta_magic_z/n, valence_p/n, year
    ).to(args.device)
    print("Model loaded successfully.")

    # --- Load Validation Data ---
    print(f"Loading validation data from: {val_path}")
    val_df = pd.read_feather(val_path)
    print(f"Validation data loaded. Total samples: {len(val_df)}")
    
    # Identify encoder feature columns (must match training)
    encoder_feature_cols = [
        'Z_int', 'A_int', 'asym', 
        'is_odd_Z', 'is_odd_N', 
        'delta_magic_z', 'delta_magic_n', 
        'valence_p', 'valence_n',
        'year'
    ]


    # --- Generate and Print Predictions ---
    print(f"\n--- Generating predictions for {args.n_samples} random samples from the validation set ---\n")
    
    sample_df = val_df.sample(n=args.n_samples, random_state=cfg.get('seed', 42))

    for _, row in sample_df.iterrows():
        print("-" * 80)
        print(f"Predicting for nuclide (Z={row['Z_int']}, A={row['A_int']}, Year={row['year']})")
        
        encoder_features = row[encoder_feature_cols].values.astype('float32')
        
        # Run greedy search
        pred_tokens, pred_delta_es = generate_sequence(model, tokenizer, encoder_features, delta_e_mean, delta_e_std, max_len=256)
        
        reconstruct_and_print(tokenizer, pred_tokens, pred_delta_es, row)
        
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main() 