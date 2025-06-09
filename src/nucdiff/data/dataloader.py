# src/nucdiff/data/dataloader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# ─── 1) 配置三张表对应的文件名 + 它们各自的目标列 ───────────
RECORD_FILES = {
    "L": ("levels.feather", "energy_keV"),
    "G": ("gammas.feather", "e_gamma_keV"),
    "Q": ("q.feather",    "q_value_keV"),
}

class ENSDFDataset(Dataset):
    def __init__(self, df, numeric_cols, elem2idx, rec2idx):
        # —— 1. 数值特征矩阵 [N, D]，NaN→0
        arr = df[numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        self.X_num = torch.from_numpy(arr)
        self.numeric_dim = len(numeric_cols)

        # —— 2. 类别特征索引
        self.X_elem = torch.tensor(df["element_idx"].to_numpy(), dtype=torch.long)
        self.X_rec  = torch.tensor(df["record_type_idx"].to_numpy(), dtype=torch.long)

        # —— 3. 目标值张量
        self.y = torch.tensor(df["target"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            {
                "num":  self.X_num[idx],
                "elem": self.X_elem[idx],
                "rec":  self.X_rec[idx],
            },
            self.y[idx]
        )

def build_dataset(year: int, cfg: dict):
    # ─── 1) 依次读取三张表、打上 record_type & target ───────────
    parts = []
    for rec, (fname, tcol) in RECORD_FILES.items():
        df = pd.read_feather(fname)
        # 只保留指定年份；dataset_year 列已在 parser.py 中生成
        df = df[df["dataset_year"] == year].copy()
        # 统一列
        df["record_type"] = rec
        df["target"]       = df[tcol]
        parts.append(df)

    # ─── 2) 合并所有记录 ─────────────────────────────────────
    df_all = pd.concat(parts, ignore_index=True)

    # ─── 3) 处理类别特征，确保全是 str → 索引 ────────────────
    # element 列仅在 Q 表有，其它表填 "UNK"
    df_all["element"] = df_all.get("element", pd.NA).fillna("UNK").astype(str)
    df_all["element"] = df_all["element"].fillna("UNK").astype(str)
    # 建映射：element → idx；sorted 只在同类型 str 间比较，无错误
    elems = sorted(df_all["element"].unique())
    elem2idx = {e: i for i, e in enumerate(elems)}
    df_all["element_idx"] = df_all["element"].map(elem2idx)

    # record_type → idx
    recs = ["L", "G", "Q"]
    rec2idx = {r: i for i, r in enumerate(recs)}
    df_all["record_type_idx"] = df_all["record_type"].map(rec2idx)

    # ─── 4) 自动挑出所有数值列，去掉不当特征 ────────────────
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    # 删掉年份 & 目标列，它们不是输入
    num_cols = [c for c in num_cols if c not in ("dataset_year", "target")]

    numeric_dim = len(num_cols)    

    # ─── 5) 划分 train / val（80/20）──────────────────────
    n_cut = int(len(df_all) * cfg.get("train_frac", 0.8))
    df_tr, df_va = df_all.iloc[:n_cut], df_all.iloc[n_cut:]

    # ─── 6) 返回 Dataset + 映射字典 ────────────────────────
    train_ds = ENSDFDataset(df_tr, num_cols, elem2idx, rec2idx)
    val_ds   = ENSDFDataset(df_va, num_cols, elem2idx, rec2idx)
    return train_ds, val_ds, (elem2idx, rec2idx, numeric_dim)

# —— 额外 helper：一行拿到 DataLoader —— 
def get_loaders(year: int, cfg: dict):
    train_ds, val_ds, maps = build_dataset(year, cfg)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"]
    )
    return train_loader, val_loader, maps
