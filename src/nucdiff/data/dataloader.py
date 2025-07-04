# src/nucdiff/data/dataloader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# --- 1) Configure file names and target columns for the three tables ---
RECORD_FILES = {
    "L": ("levels.feather", "energy_keV"),
    "G": ("gammas.feather", "e_gamma_keV"),
    "Q": ("q.feather", "q_value_keV"),
}


class ENSDFDataset(Dataset):
    def __init__(self, df, numeric_cols, elem2idx, rec2idx):
        # 1. Numeric feature matrix [N, D], NaN→0
        arr = df[numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        self.X_num = torch.from_numpy(arr)
        self.numeric_dim = len(numeric_cols)

        # 2. Categorical feature indices
        self.X_elem = torch.tensor(df["element_idx"].to_numpy(), dtype=torch.long)
        self.X_rec = torch.tensor(df["record_type_idx"].to_numpy(), dtype=torch.long)

        # 3. Multi-task target tensors
        self.y_L = torch.tensor(df["target_L"].to_numpy(), dtype=torch.float32)
        self.y_G = torch.tensor(df["target_G"].to_numpy(), dtype=torch.float32)
        self.y_Q = torch.tensor(df["target_Q"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.y_L)

    def __getitem__(self, idx):
        return (
            {
                "numeric": self.X_num[idx],
                "element": self.X_elem[idx],
                "record_type": self.X_rec[idx],
            },
            {
                "L": self.y_L[idx],
                "G": self.y_G[idx],
                "Q": self.y_Q[idx],
            }
        )


def build_dataset(year: int, cfg: dict):
    # 1) Read all three tables, add record_type & target
    parts = []
    for rec, (fname, tcol) in RECORD_FILES.items():
        df = pd.read_feather(fname)
        # Only keep the specified year; dataset_year column is generated in parser.py
        df = df[df["dataset_year"] == year].copy()
        # Standardize columns
        df["record_type"] = rec
        df[f"target_{rec}"] = df[tcol]
        parts.append(df)

    # 2) Concatenate all records
    df_all = pd.concat(parts, ignore_index=True)

    # 3) Process categorical features, ensure all are str → index
    # element column only exists in Q table, others fill with "UNK"
    element_col = df_all.get("element")
    if element_col is not None:
        df_all["element"] = element_col.fillna("UNK").astype(str)
    else:
        df_all["element"] = "UNK"
    df_all["element"] = df_all["element"].fillna("UNK").astype(str)
    # Build mapping: element → idx; sorted only compares same type str, no error
    elems = sorted(df_all["element"].unique())
    elem2idx = {e: i for i, e in enumerate(elems)}
    df_all["element_idx"] = df_all["element"].map(lambda x: elem2idx[x])

    # record_type → idx
    recs = ["L", "G", "Q"]
    rec2idx = {r: i for i, r in enumerate(recs)}
    df_all["record_type_idx"] = df_all["record_type"].map(lambda x: rec2idx[x])

    # 4) Automatically select all numeric columns, remove inappropriate features
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    # Remove year & target columns, they are not inputs
    num_cols = [c for c in num_cols if c not in ("dataset_year", "target_L", "target_G", "target_Q")]

    numeric_dim = len(num_cols)

    # 5) Split train / val (80/20)
    n_cut = int(len(df_all) * cfg.get("train_frac", 0.8))
    df_tr, df_va = df_all.iloc[:n_cut], df_all.iloc[n_cut:]

    # 6) Return Dataset + mapping dicts
    train_ds = ENSDFDataset(df_tr, num_cols, elem2idx, rec2idx)
    val_ds = ENSDFDataset(df_va, num_cols, elem2idx, rec2idx)
    return train_ds, val_ds, (elem2idx, rec2idx, numeric_dim)


# Extra helper: get DataLoader in one line
def get_loaders(year: int, cfg: dict):
    train_ds, val_ds, maps = build_dataset(year, cfg)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"])
    return train_loader, val_loader, maps
