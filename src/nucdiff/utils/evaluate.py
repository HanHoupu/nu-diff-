import torch, numpy as np

def evaluate_mae(model, loader):
    model.eval(); mae = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            mae.append(torch.abs(pred - y).cpu().numpy())
    return float(np.concatenate(mae).mean())
