"""
Parse *.Q.ens / *.L.ens / *.G.ens in the specified directory
and save each as a Feather file
----------------------------------------------
Usage:
    python parse_to_feather.py <data directory>
    # If no argument is given, use current directory by default
Output:
    q.feather        – Q records
    levels.feather   – L records
    gammas.feather   – G records
"""

import sys, json
from pathlib import Path
import pandas as pd

from parser import (  #
    iter_q_records,
    iter_l_records,
    iter_g_records,
)

# ---------- select data root directory ----------
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

# ---------- 1) parse Q file ----------
q_records = []
for fp in root.rglob("*.Q"):
    q_records.extend(iter_q_records(fp))

# ---------- 2) parse L file ----------
l_records = []
for fp in root.rglob("*.L"):
    l_records.extend(iter_l_records(fp))

# -- build "energy → level_id" index (rounded to 1 keV)
level_idx = {
    round(l.energy_keV, 1): l.level_id for l in l_records if l.energy_keV is not None
}

# ---------- 3) parse G file ----------
g_records = []
for fp in root.rglob("*.G"):
    g_records.extend(iter_g_records(fp, level_idx))


# ---------- 4) convert to DataFrame and save as Feather ----------
def to_df(objs):
    """Convert dataclass list to DataFrame, and convert extras dict to JSON string"""
    df = pd.DataFrame([o.__dict__ for o in objs])
    if "extras" in df.columns:
        df["extras"] = df["extras"].apply(json.dumps)
    return df


to_df(q_records).to_feather("q.feather")
to_df(l_records).to_feather("levels.feather")
to_df(g_records).to_feather("gammas.feather")

print("✔  Generated  q.feather / levels.feather / gammas.feather")
