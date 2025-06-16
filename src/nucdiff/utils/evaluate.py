import torch, numpy as np

import torch, numpy as np


def evaluate_mae(model, loader):
    model.eval()
    device = next(model.parameters()).device  # ← 取模型当前设备
    mae_list = []
    with torch.no_grad():
        for x, y in loader:
            # 把 batch 全搬到同一设备
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)

            pred = model(x)
            mae_list.append(torch.abs(pred - y).cpu().numpy())

    return float(np.concatenate(mae_list).mean())
