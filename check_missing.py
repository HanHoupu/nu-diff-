# check_missing.py
import pandas as pd
from pathlib import Path

# ---------- 1. 读取三张表 ----------
tables = {
    "levels": Path("levels.feather"),
    "gammas": Path("gammas.feather"),
    "q": Path("q.feather"),
}

dfs = {name: pd.read_feather(path) for name, path in tables.items()}

# ---------- 2. 统计缺失 & 打印示例 ----------
for name, df in dfs.items():
    print(f"\n=== {name.upper()}  ({len(df)} rows, {df.shape[1]} cols) ===")

    for col in df.columns:
        null_cnt = df[col].isna().sum()
        null_pct = null_cnt / len(df) * 100

        # 找到首个非缺失示例
        sample_val = None
        sample_row = None
        if null_cnt < len(df):
            first_idx = df[col].first_valid_index()
            sample_val = df.at[first_idx, col]
            sample_row = df.loc[first_idx].to_dict()

        print(f"\n• {col:>15}  " f"null: {null_cnt:>6}  " f"({null_pct:5.1f}%)")

        if sample_val is not None:
            print(f"  └ sample value: {sample_val}")
            print(f"    sample row : {sample_row}")
        else:
            print("  └ **All lost**")


print(df.head())
