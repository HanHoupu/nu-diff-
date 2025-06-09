"""
解析指定目录下的 *.Q.ens / *.L.ens / *.G.ens
并各自保存为 Feather 文件
----------------------------------------------
用法：
    python parse_to_feather.py <数据目录>
    # 如果不写参数，就默认为当前目录
生成：
    q.feather        – Q 记录
    levels.feather   – L 记录
    gammas.feather   – G 记录
"""
import sys, json
from pathlib import Path
import pandas as pd

from parser_lgq import (           # 
    iter_q_records,
    iter_l_records,
    iter_g_records,
)

# ---------- 选择数据根目录 ----------
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

# ---------- 1) 解析 Q 文件 ----------
q_records = []
for fp in root.rglob("*.Q.ens"):
    q_records.extend(iter_q_records(fp))

# ---------- 2) 解析 L 文件 ----------
l_records = []
for fp in root.rglob("*.L.ens"):
    l_records.extend(iter_l_records(fp))

# —— 建立 “能量 → level_id” 索引（四舍五入到 1 keV）
level_idx = {round(l.energy_keV, 1): l.level_id
             for l in l_records if l.energy_keV is not None}

# ---------- 3) 解析 G 文件 ----------
g_records = []
for fp in root.rglob("*.G.ens"):
    g_records.extend(iter_g_records(fp, level_idx))

# ---------- 4) 转 DataFrame 并保存 Feather ----------
def to_df(objs):
    """把 dataclass 列表转成 DataFrame，并把 extras 字典转成 JSON 字符串"""
    df = pd.DataFrame([o.__dict__ for o in objs])
    if "extras" in df.columns:
        df["extras"] = df["extras"].apply(json.dumps)
    return df

to_df(q_records).to_feather("q.feather")
to_df(l_records).to_feather("levels.feather")
to_df(g_records).to_feather("gammas.feather")

print("✔  已生成  q.feather / levels.feather / gammas.feather")
