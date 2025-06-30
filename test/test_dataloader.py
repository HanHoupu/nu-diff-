"""
Make sure the numeric feature dimension output by dataloader is correct
"""
import numpy as np
from nucdiff.data.dataloader import build_dataset
import torch

def test_numeric_dim():
    cfg = {"split": "train"}         # simple example, just use the cfg in your project
    ds, _, _ = build_dataset(2004, cfg)
    sample = ds[0][0]                # (batch=0, timestep=0)
    numeric = sample["numeric"]      # e.g. shape = (7,)
    assert numeric.dim() == 1        # 1D tensor
    assert numeric.dtype == torch.float32

def test_dataset_len():
    cfg = {"split": "train"}
    ds, _, _ = build_dataset(2004, cfg)
    assert len(ds) > 0
