import torch, numpy as np
from sklearn.metrics import r2_score

def evaluate_mae(model, loader, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device  # get model device now
    mae_list = []
    with torch.no_grad():
        for x, y in loader:
            # move all batch to same device
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            pred = model(x)
            mae = torch.mean(torch.abs(pred - y))
            mae_list.append(mae.item())
    return sum(mae_list) / len(mae_list)

def evaluate_rmse(model, loader, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    rmse_list = []
    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            pred = model(x)
            rmse = torch.sqrt(torch.mean((pred - y) ** 2))
            rmse_list.append(rmse.item())
    return sum(rmse_list) / len(rmse_list)

def evaluate_r2(model, loader, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            pred = model(x)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return r2_score(all_targets, all_preds)
