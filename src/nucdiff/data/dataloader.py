# src/nucdiff/data/dataloader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
from pathlib import Path
from typing import List, Dict, Iterator, Optional
import json

# --- 1) Configure file names and target columns for the three tables ---
RECORD_FILES = {
    "L": ("ENSDF-{year}.L", "energy_keV"),
    "G": ("ENSDF-{year}.G", "e_gamma_keV"),
    "Q": ("ENSDF-{year}.Q", "q_value_keV"),
}

# ---------------- common tools -----------------
NUM_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?")  # match float/scientific
KV_RE = re.compile(r"([A-Za-z%]+)\s*=?\s*([\-\w.+]+)")  # match KEY=VAL
YEAR_RE = re.compile(r"(\d{4})")


def _first_float(text: str) -> Optional[float]:
    m = NUM_RE.search(text)
    return float(m.group()) if m else None


def _year(fp: Path) -> Optional[int]:
    m = YEAR_RE.search(fp.stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- Q ----------
def iter_q_records(fp: Path) -> Iterator[Dict]:
    year = _year(fp)
    with fp.open("r", encoding="utf-8", errors="ignore") as fin:
        for ln in fin:
            cols = ln.strip().split(maxsplit=4)
            if len(cols) < 2 or cols[1] != "Q":
                continue

            # -------- Z, A, Element --------
            ZA = cols[0]
            z_match = re.match(r"\d+", ZA)
            z = int(z_match.group()) if z_match else 0
            rest = ZA[z_match.end() :] if z_match else ZA
            a_match = re.match(r"\d+", rest)
            if a_match:
                a = int(a_match.group())
                element = rest[a_match.end() :]
            else:
                a = 0
                element = rest

            q_val = _first_float(cols[2]) if len(cols) > 2 else None
            m_exc = _first_float(cols[3]) if len(cols) > 3 else None
            src = cols[4] if len(cols) > 4 else None

            yield {
                "z": z,
                "a": a,
                "element": element,
                "q_value_keV": q_val,
                "m_exc_keV": m_exc,
                "source": src,
                "dataset_year": year
            }


# ---------------------------------------------------------------- L ----------
def iter_l_records(fp: Path) -> Iterator[Dict]:
    year = _year(fp)
    buf: List[str] = []
    level_id = 0

    def flush():
        nonlocal level_id, buf
        if not buf:
            return
        main = buf[0].strip().split(maxsplit=4)
        energy = _first_float(main[2]) if len(main) > 2 else None
        jp = main[3] if len(main) > 3 else ""
        hl = main[4] if len(main) > 4 else ""
        extras = {}
        for l in buf:
            extras.update({k.upper(): v for k, v in KV_RE.findall(l)})
        level_id += 1
        rec = {
            "level_id": level_id,
            "energy_keV": energy,
            "jp": jp,
            "half_life": hl,
            "extras": json.dumps(extras),
            "dataset_year": year
        }
        buf.clear()
        return rec

    for ln in fp.open("r", encoding="utf-8", errors="ignore"):
        parts = ln.strip().split(maxsplit=2)
        flag = parts[1] if len(parts) > 1 else ""
        if flag == "L":
            r = flush()
            if r:
                yield r
        if flag.startswith("L"):
            buf.append(ln)
    r = flush()
    if r:
        yield r


# ---------------------------------------------------------------- G ----------
def iter_g_records(fp: Path) -> Iterator[Dict]:
    year = _year(fp)
    buf: List[str] = []

    def flush():
        if not buf:
            return

        main = buf[0].strip().split(maxsplit=6)
        e_g = _first_float(main[2]) if len(main) > 2 else None
        inten = main[3] if len(main) > 3 else None
        mul = main[5] if len(main) > 5 else None

        g = {
            "e_gamma_keV": e_g,
            "intensity": inten,
            "multipolarity": mul,
            "dataset_year": year,
            "width_ev": None,
            "width_unc_ev": None,
            "bm1w": None,
            "be2w": None,
            "extras": "{}",
            "from_id": None,
            "to_id": None
        }

        for l in buf:
            tail = l.strip()

            # WIDTHG
            if m := re.search(r"WIDTHG\s*=?\s*([-\d.E+]+)\s*EV\s*([\d.]*)", tail, re.I):
                g["width_ev"] = _first_float(m.group(1))
                g["width_unc_ev"] = _first_float(m.group(2))

            # BM1W / BE2W
            if m := re.search(r"BM1W\s*=?\s*([-\d.E+]+)", tail, re.I):
                g["bm1w"] = _first_float(m.group(1))
            if m := re.search(r"BE2W\s*=?\s*([-\d.E+]+)", tail, re.I):
                g["be2w"] = _first_float(m.group(1))

            # 其他键值
            extras = {}
            for k, v in KV_RE.findall(tail):
                if k.upper() not in ("WIDTHG", "BM1W", "BE2W"):
                    extras[k.upper()] = v
            if extras:
                g["extras"] = json.dumps(extras)

        buf.clear()
        return g

    # -------------- 逐行读取并收集块 ----------------
    for ln in fp.open("r", encoding="utf-8", errors="ignore"):
        cols = ln.strip().split(maxsplit=2)
        flag = cols[1] if len(cols) > 1 else ""
        if flag == "G":
            r = flush()
            if r:
                yield r
        if flag.startswith("G"):
            buf.append(ln)
    r = flush()
    if r:
        yield r


class ENSDFDataset(Dataset):
    def __init__(self, df, numeric_cols, elem2idx, rec2idx, is_train=True, stats=None):
        # 1. Numeric feature matrix [N, D], NaN→0
        arr = df[numeric_cols].fillna(0).to_numpy(dtype=np.float32)
        self.X_num = torch.from_numpy(arr)
        self.numeric_dim = len(numeric_cols)

        # 2. Categorical feature indices
        self.X_elem = torch.tensor(df["element_idx"].to_numpy(), dtype=torch.long)
        self.X_rec = torch.tensor(df["record_type_idx"].to_numpy(), dtype=torch.long)

        # 3. Multi-task target tensors with normalization
        y_L_raw = torch.tensor(df["target_L"].to_numpy(), dtype=torch.float32)
        y_G_raw = torch.tensor(df["target_G"].to_numpy(), dtype=torch.float32)
        y_Q_raw = torch.tensor(df["target_Q"].to_numpy(), dtype=torch.float32)
        
        if is_train:
            # 计算训练集的统计信息（忽略NaN值）
            self.L_mean = torch.nanmean(y_L_raw)
            self.L_std = torch.sqrt(torch.nanmean((y_L_raw - self.L_mean) ** 2))
            self.G_mean = torch.nanmean(y_G_raw)
            self.G_std = torch.sqrt(torch.nanmean((y_G_raw - self.G_mean) ** 2))
            self.Q_mean = torch.nanmean(y_Q_raw)
            self.Q_std = torch.sqrt(torch.nanmean((y_Q_raw - self.Q_mean) ** 2))
            
            # 避免除零
            self.L_std = self.L_std if self.L_std > 0 else 1.0
            self.G_std = self.G_std if self.G_std > 0 else 1.0
            self.Q_std = self.Q_std if self.Q_std > 0 else 1.0
        else:
            # 使用传入的统计信息
            if stats is not None:
                self.L_mean, self.L_std = stats['L_mean'], stats['L_std']
                self.G_mean, self.G_std = stats['G_mean'], stats['G_std']
                self.Q_mean, self.Q_std = stats['Q_mean'], stats['Q_std']
            else:
                # 如果没有统计信息，使用默认值
                self.L_mean, self.L_std = 0.0, 1.0
                self.G_mean, self.G_std = 0.0, 1.0
                self.Q_mean, self.Q_std = 0.0, 1.0
        
        # 标准化标签（处理NaN值）
        self.y_L = torch.where(torch.isnan(y_L_raw), 
                              torch.nan, 
                              (y_L_raw - self.L_mean) / self.L_std)
        self.y_G = torch.where(torch.isnan(y_G_raw), 
                              torch.nan, 
                              (y_G_raw - self.G_mean) / self.G_std)
        self.y_Q = torch.where(torch.isnan(y_Q_raw), 
                              torch.nan, 
                              (y_Q_raw - self.Q_mean) / self.Q_std)

    def __len__(self):
        return len(self.y_L)

    def __getitem__(self, idx):
        # Generate feature_type_ids for each numeric feature
        feature_type_ids = torch.arange(self.numeric_dim, dtype=torch.long)
        
        return (
            {
                "numeric": self.X_num[idx],
                "element": self.X_elem[idx],
                "record_type": self.X_rec[idx],
                "feature_type_ids": feature_type_ids,
            },
            {
                "L": self.y_L[idx],
                "G": self.y_G[idx],
                "Q": self.y_Q[idx],
            }
        )


def build_dataset(year: int, cfg: dict):
    # 1) Read all three tables from clean data folder
    clean_data_path = Path("clean data")
    parts = []
    
    for rec, (fname_pattern, tcol) in RECORD_FILES.items():
        fname = fname_pattern.format(year=year)
        file_path = clean_data_path / fname
        
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping {rec} records")
            continue
            
        # Parse records based on type
        if rec == "Q":
            records = list(iter_q_records(file_path))
        elif rec == "L":
            records = list(iter_l_records(file_path))
        elif rec == "G":
            records = list(iter_g_records(file_path))
        
        if records:
            df = pd.DataFrame(records)
            # Standardize columns
            df["record_type"] = rec
            df[f"target_{rec}"] = df[tcol]
            parts.append(df)

    if not parts:
        raise ValueError(f"No data found for year {year}")

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
    train_ds = ENSDFDataset(df_tr, num_cols, elem2idx, rec2idx, is_train=True)
    
    # 获取训练集的统计信息
    stats = {
        'L_mean': train_ds.L_mean,
        'L_std': train_ds.L_std,
        'G_mean': train_ds.G_mean,
        'G_std': train_ds.G_std,
        'Q_mean': train_ds.Q_mean,
        'Q_std': train_ds.Q_std,
    }
    
    val_ds = ENSDFDataset(df_va, num_cols, elem2idx, rec2idx, is_train=False, stats=stats)
    return train_ds, val_ds, (elem2idx, rec2idx, numeric_dim)


# Extra helper: get DataLoader in one line
def get_loaders(year: int, cfg: dict):
    train_ds, val_ds, maps = build_dataset(year, cfg)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg["batch_size"])
    return train_loader, val_loader, maps
