import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
import numpy as np
import os
import json
import sys
import heapq

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.data.tokenizer import LevelTokenizer
from src.nucdiff.train import NucDiffSeq2SeqModule
from src.nucdiff.utils.paths import get_preprocessed_paths

def beam_search_generate(model, tokenizer, encoder_features, delta_e_mean, delta_e_std, beam_width=5, max_len=50):
    """
    Generates a sequence using the trained model with beam search, adapted for the hybrid output.
    """
    model.eval()
    device = next(model.parameters()).device
    
    encoder_features = torch.tensor(encoder_features, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Start with a beam containing one hypothesis: ([SOS], [dE=0], score=0.0)
    # The score is the log probability of the sequence.
    initial_dE_norm = (0.0 - delta_e_mean) / (delta_e_std + 1e-6)
    beams = [([tokenizer.sos_id], [initial_dE_norm], [0.0], 0.0)] # (tokens, normalized_dEs, keV_dEs, log_prob)

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for tokens, dEs_norm, dEs_keV, score in beams:
                # If the last token is EOS, this beam is finished.
                if tokens[-1] == tokenizer.eos_id:
                    heapq.heappush(new_beams, (score, (tokens, dEs_norm, dEs_keV, score)))
                    continue

                input_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
                input_delta_e_norm = torch.tensor([dEs_norm], dtype=torch.float32, device=device)
                
                class_logits, regr_output_norm = model.model(
                    src=encoder_features,
                    tgt_tokens=input_tokens,
                    tgt_delta_e=input_delta_e_norm,
                    tgt_padding_mask=None
                )
                
                class_log_probs = F.log_softmax(class_logits[:, -1, :], dim=-1)
                topk_log_probs, topk_ids = torch.topk(class_log_probs, beam_width, dim=-1)
                
                for i in range(beam_width):
                    next_token_id = topk_ids[0, i].item()
                    log_prob = topk_log_probs[0, i].item()
                    
                    new_tokens = tokens + [next_token_id]
                    
                    # The regression output corresponds to the prediction for the *next* step
                    next_dE_norm = regr_output_norm[:, -1].item()
                    next_dE_keV = max(0, (next_dE_norm * delta_e_std) + delta_e_mean)
                    
                    new_dEs_norm = dEs_norm + [(next_dE_keV - delta_e_mean) / (delta_e_std + 1e-6)]
                    new_dEs_keV = dEs_keV + [next_dE_keV]
                    
                    new_score = score + log_prob
                    
                    # Use negative score because heapq is a min-heap
                    heapq.heappush(new_beams, (-new_score, (new_tokens, new_dEs_norm, new_dEs_keV, new_score)))

            # Prune the beams to keep only the top `beam_width` hypotheses
            beams = [item[1] for item in heapq.nsmallest(beam_width, new_beams)]
            
            # Check if all beams have ended in EOS
            if all(b[0][-1] == tokenizer.eos_id for b in beams):
                break
    
    # Return the best hypothesis (highest log probability)
    best_beam = max(beams, key=lambda x: x[3])
    return best_beam[0], best_beam[2] # Return tokens and keV_dEs


def reconstruct_and_print(tokenizer, tokens, delta_es, true_row):
    """Helper to print truth vs predicted."""
    pred_energy = 0
    pred_parts = []
    for i in range(1, len(tokens)): # Skip SOS token
        if tokens[i] == tokenizer.eos_id: break
        pred_energy += delta_es[i]
        token_str = tokenizer.id_to_token.get(tokens[i], "[UNK]")
        pred_parts.append(f"E={pred_energy:.3f}, {token_str}")
    predicted_sequence = "; ".join(pred_parts)

    true_delta_e = json.loads(true_row['delta_e_sequence'])
    true_tokens = true_row['token_sequence'].split(';')
    true_energy = 0
    true_parts = []
    for i, token in enumerate(true_tokens):
        true_energy += true_delta_e[i]
        true_parts.append(f"E={true_energy:.3f}, {token}")
    true_sequence = "; ".join(true_parts)

    print(f"\n[TRUTH]   : {true_sequence}")
    print(f"[PREDICTED]: {predicted_sequence}")

def main():
    parser = argparse.ArgumentParser(description="Generate sequences from a trained Seq2Seq model.")
    parser.add_argument("--cfg", type=Path, default=PROJECT_ROOT / "configs/temporal.yaml", help="Path to the model config file.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the trained model checkpoint. If None, finds the latest one.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to predict from the validation set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (cuda or cpu).")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for beam search decoding.")
    args = parser.parse_args()

    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    path_cfg = {
        "seed": cfg["seed"], "val_frac": cfg["val_frac"],
        "base_start_year": cfg["base_start_year"], "base_end_year": cfg["base_end_year"]
    }
    _, val_path = get_preprocessed_paths(path_cfg, 'SEQ')
    preprocessed_dir = val_path.parent

    tokenizer = LevelTokenizer.load(preprocessed_dir / "tokenizer_vocab.json")
    with open(preprocessed_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    delta_e_scaler_info = metadata.get('delta_e_scaler', {'mean': 0.0, 'std': 1.0})
    delta_e_mean = delta_e_scaler_info['mean']
    delta_e_std = delta_e_scaler_info['std']

    # FIXED: Use centralized extended feature column definition
    from src.nucdiff.utils.constants import ENCODER_FEATURE_COLS_EXTENDED
    encoder_feature_cols = ENCODER_FEATURE_COLS_EXTENDED

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        print("No checkpoint path provided. Finding the latest best model...")
        run_dir = PROJECT_ROOT / "runs"
        potential_prefixes = ['temporal_SEQ_', 'pl_SEQ_']
        candidate_dirs = []
        for prefix in potential_prefixes:
            candidate_dirs.extend([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)])
        if not candidate_dirs:
            raise FileNotFoundError(f"No run directories with prefixes {potential_prefixes} found in {run_dir}.")
        latest_run = sorted(candidate_dirs, key=os.path.getmtime, reverse=True)[0]
        print(f"Found latest run directory: {latest_run.name}")
        checkpoint_dir = latest_run
        if latest_run.name.startswith('pl_'):
            version_dirs = sorted([d for d in latest_run.iterdir() if d.is_dir() and d.name.startswith('version_')], key=os.path.getmtime, reverse=True)
            if version_dirs:
                checkpoint_dir = version_dirs[0]
        checkpoint_dir = checkpoint_dir / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("best-*.ckpt"))
        if not checkpoint_files:
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}.")
        checkpoint_path = sorted(checkpoint_files, key=os.path.getmtime, reverse=True)[0]

    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = NucDiffSeq2SeqModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_id, delta_e_mean=delta_e_mean,
        delta_e_std=delta_e_std, encoder_feature_dim=len(encoder_feature_cols)
    ).to(args.device)
    print("Model loaded successfully.")

    print(f"Loading validation data from: {val_path}")
    val_df = pd.read_feather(val_path)
    
    print(f"\n--- Generating predictions for {args.n_samples} random samples (Beam Width: {args.beam_width}) ---\n")
    sample_df = val_df.sample(n=args.n_samples, random_state=cfg.get('seed', 42))

    for _, row in sample_df.iterrows():
        print("-" * 80)
        print(f"Predicting for nuclide (Z={row['Z_int']}, A={row['A_int']}, Year={row['year']})")
        encoder_features = row[encoder_feature_cols].values.astype('float32')
        pred_tokens, pred_delta_es = beam_search_generate(
            model, tokenizer, encoder_features, delta_e_mean, delta_e_std, beam_width=args.beam_width
        )
        reconstruct_and_print(tokenizer, pred_tokens, pred_delta_es, row)
        print("-" * 80 + "\n")

if __name__ == "__main__":
    main()
