# tests/test_model.py  —— 极简、纯虚拟数据版
# tests/test_model.py  -- super simple, just use fake data
import torch
from nucdiff.model.incremental import IncrementalModel


def test_forward_backward_no_nan():
    numeric_dim = 15           # 
    batch_size = 2

    # 
    model = IncrementalModel(
        elem2idx={"H": 0},
        rec2idx={"L": 0},
        numeric_dim=numeric_dim,
    ).train()

    # 
    batch = {
        "num":  torch.randn(batch_size, numeric_dim),   # (B, D) float32
        "elem": torch.zeros(batch_size, dtype=torch.long),  # (B,) long
        "rec":  torch.zeros(batch_size, dtype=torch.long),  # (B,) long
    }


    output = model(batch)
    loss = output.pow(2).mean()
    assert not torch.isnan(loss)

    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
