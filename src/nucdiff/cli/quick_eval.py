#!/usr/bin/env python3
"""
Quick evaluation script for saved checkpoints.
Example:
  python -m nucdiff.cli.quick_eval --ckpt checkpoints/2023.pt --year 2023 --cfg configs/default.yaml
"""
import pathlib, sys
import argparse, datetime, csv, yaml, torch
from torch.utils.data import DataLoader

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from nucdiff.data.dataloader import build_dataset
from nucdiff.model.incremental import IncrementalModel
from nucdiff.utils.evaluate import evaluate_mae, evaluate_rmse, evaluate_r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--year", type=int, required=True, help="Year for evaluation")
    parser.add_argument("--cfg", type=str, default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not pathlib.Path(args.ckpt).exists():
        print(f"Error: Checkpoint file {args.ckpt} not found")
        sys.exit(1)
    
    # Load config
    try:
        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Build dataset and dataloader
    try:
        train_ds, val_ds, (elem2idx, rec2idx, numeric_dim) = build_dataset(args.year, cfg)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    except Exception as e:
        print(f"Error building dataset: {e}")
        sys.exit(1)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_backbone = cfg["train_backbone_first_year"] and args.year == cfg["start_year"]
    model = IncrementalModel(
        elem2idx=elem2idx,
        rec2idx=rec2idx,
        numeric_dim=numeric_dim,
        rank=cfg["rank"],
        alpha=cfg["alpha"],
        embed_dim=cfg["embed_dim"],
        train_backbone=train_backbone,
    ).to(device)
    
    # Load checkpoint
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Evaluate
    try:
        mae = evaluate_mae(model, val_loader, device)
        rmse = evaluate_rmse(model, val_loader, device)
        r2 = evaluate_r2(model, val_loader, device)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
    
    # Print results
    print(f"{args.ckpt} | MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    
    # Save to CSV
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    csv_path = logs_dir / "metrics.csv"
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [timestamp, args.ckpt, args.year, f"{mae:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
    
    # Write CSV header if file doesn't exist
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "checkpoint", "year", "mae", "rmse", "r2"])
    
    # Append results
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
    
    print(f"Results saved to: {csv_path}")

if __name__ == "__main__":
    main() 