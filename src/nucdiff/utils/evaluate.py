import torch, numpy as np

import torch, numpy as np


def evaluate_mae(model, loader):
    model.eval()
    device = next(model.parameters()).device  # get model device now
    mae_list = []
    with torch.no_grad():
        for x, y in loader:
            # move all batch to same device
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)

            pred = model(x)
            mae_list.append(torch.abs(pred - y).cpu().numpy())

    return float(np.concatenate(mae_list).mean())
