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

# incremental LoRA main model
from nucdiff.model.incremental import IncrementalModel
from nucdiff.utils.seed import set_seed
from nucdiff.utils.earlystop import EarlyStopper
from nucdiff.utils.fisher import fisher_l2_reg
from nucdiff.utils.evaluate import evaluate_mae
from nucdiff.data.dataloader import build_dataset
from nucdiff.model.safety import SafetyCallback 

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

# --- optimizer only update LoRA and head ---
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=float(cfg["lr"]),  # ← 即使 YAML 手滑写成 "1e-4" 也能转
)

# --- training loop ---
early_stopper = EarlyStopper(patience=cfg["early_stop_patience"])
for epoch in range(cfg["max_epochs"]):
    model.train()
    for batch_x, batch_y in train_loader:
        # move to device
        batch_x = {k: v.to(device) for k, v in batch_x.items()}
        batch_y = batch_y.to(device)
        # forward + loss
        loss_task = model.training_step((batch_x, batch_y))
        loss_reg = fisher_l2_reg(model, cfg.get("fisher_l2", 0.0))
        loss_total = loss_task + loss_reg
        optimizer.step()
        safety(loss_total, model) 
        optimizer.zero_grad()

    # validation
    mae = evaluate_mae(model, val_loader)
    print(f"[{args.year}] epoch {epoch} | val MAE = {mae:.4f}")
    if early_stopper.step(mae):
        print("Early stop triggered.")
        break

# --- save backbone + this year LoRA ---
torch.save(model.state_dict(), run_dir / "checkpoints" / f"model_{args.year}.pth")
model.save_lora(str(run_dir / "checkpoints" / f"lora_{args.year}.pth"))

print(f"✔ Year {args.year} finished. Artifacts saved to: {run_dir}")
