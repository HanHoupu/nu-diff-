#!/usr/bin/env python3
"""
Train entry. Example:
  python -m nucdiff.train --year 2004 --cfg configs/default.yaml
"""
import pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import argparse, datetime, subprocess, shutil, yaml, torch
from torch.utils.data import DataLoader

# new DataLoader, L/G/Q three tables together
from nucdiff.data.dataloader import get_loaders, build_dataset

# Transformer multi-task model
from nucdiff.model import TransformerModel
from nucdiff.utils.seed import set_seed
from nucdiff.utils.earlystop import EarlyStopper
from nucdiff.utils.fisher import fisher_l2_reg
from nucdiff.utils.evaluate import evaluate_mae_multi
from nucdiff.data.dataloader import build_dataset
from nucdiff.model.safety import SafetyCallback 
from nucdiff.utils.logger import TrainLogger

safety = SafetyCallback(clip_norm=1.0)   

parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--cfg", type=str, default="configs/default.yaml")
args = parser.parse_args()

# --- read config & set random seed ---
with open(args.cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
set_seed(cfg["seed"])

# --- make run dir & save meta info ---
now = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
run_dir = PROJECT_ROOT / f"outputs/run-{now}"
(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(run_dir / "logs").mkdir(exist_ok=True)

# Git commit
try:
    commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
        .decode()
        .strip()
    )
except subprocess.SubprocessError:
    commit = "no-git-found"
(run_dir / "commit.txt").write_text(commit + "\n")
# dependency list
reqs = subprocess.check_output(["pip", "freeze"]).decode()
(run_dir / "requirements.txt").write_text(reqs)
# backup config & command
shutil.copy(args.cfg, run_dir / "config.yaml")
(run_dir / "cmd.txt").write_text(" ".join(sys.argv) + "\n")

# --- build dataset & DataLoader ---
train_ds, val_ds, (elem2idx, rec2idx, numeric_dim) = build_dataset(args.year, cfg)
train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

# --- device & model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(cfg, elem2idx, rec2idx).to(device)

# --- optimizer only update LoRA and head ---
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=float(cfg["lr"]),  # ← 即使 YAML 手滑写成 "1e-4" 也能转
)

# --- training loop ---
early_stopper = EarlyStopper(patience=cfg["early_stop_patience"])
logger = TrainLogger(cfg)
global_step = 0
log_every = cfg.get("log_every", 10)

for epoch in range(cfg["max_epochs"]):
    model.train()
    for batch_x, batch_y in train_loader:
        # move to device
        batch_x = {k: v.to(device) for k, v in batch_x.items()}
        batch_y = {k: v.to(device) for k, v in batch_y.items()}
        # forward + loss
        loss_task = model.training_step((batch_x, batch_y))
        loss_reg = fisher_l2_reg(model, cfg.get("fisher_l2", 0.0))
        loss_total = loss_task + loss_reg
        optimizer.step()
        safety(loss_total, model) 
        optimizer.zero_grad()
        
        if global_step % log_every == 0:
            logger.log_scalar("train/loss", loss_task.item() if hasattr(loss_task, 'item') else loss_task, global_step)
            logger.log_scalar("train/loss_reg", loss_reg.item() if hasattr(loss_reg, 'item') else loss_reg, global_step)
        global_step += 1

    # validation
    metrics = evaluate_mae_multi(model, val_loader, device)
    mae_avg = metrics["avg"]
    print(f"[{args.year}] epoch {epoch} | val MAE = {mae_avg:.4f} (L:{metrics['L']:.4f} G:{metrics['G']:.4f} Q:{metrics['Q']:.4f})")
    
    # Log individual task metrics
    logger.log_scalar("val/mae_L", metrics["L"], epoch)
    logger.log_scalar("val/mae_G", metrics["G"], epoch)
    logger.log_scalar("val/mae_Q", metrics["Q"], epoch)
    logger.log_scalar("val/mae_avg", mae_avg, epoch)
    
    if early_stopper.step(mae_avg):
        print("Early stop triggered.")
        break

# --- save backbone + this year LoRA ---
torch.save(model.state_dict(), run_dir / "checkpoints" / f"model_{args.year}.pth")
model.save_lora(str(run_dir / "checkpoints" / f"lora_{args.year}.pth"))

logger.close()
print(f"✔ Year {args.year} finished. Artifacts saved to: {run_dir}")
