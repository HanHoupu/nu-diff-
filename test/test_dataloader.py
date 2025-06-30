"""
Make sure the numeric feature dimension output by dataloader is correct
"""
import numpy as np
from nucdiff.data.dataloader import build_dataset

def test_numeric_dim():
    cfg = {"split": "train"}         # simple example, just use the cfg in your project
    ds, _, _ = build_dataset(2004, cfg)
    sample = ds[0][0]                # (batch=0, timestep=0)
    numeric = sample["num"]          # e.g. shape = (7,)
    assert numeric.shape[-1] > 0
    assert not np.isnan(numeric).any()

def test_dataset_len():
    ds, _, _ = build_dataset(2004, cfg={"split": "val"})
    assert len(ds) > 0, "Dataset must not be empty"
