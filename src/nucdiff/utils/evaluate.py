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

def evaluate_mae_multi(model, loader, device=None):
    """Evaluate MAE for multi-task model (L/G/Q)"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    mae_L, mae_G, mae_Q = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x = {k: v.to(device) for k, v in x.items()}
            y = {k: v.to(device) for k, v in y.items()}
            pred = model(x)
            
            mae_L.append(torch.mean(torch.abs(pred["L"] - y["L"])).item())
            mae_G.append(torch.mean(torch.abs(pred["G"] - y["G"])).item())
            mae_Q.append(torch.mean(torch.abs(pred["Q"] - y["Q"])).item())
    
    avg_mae_L = sum(mae_L) / len(mae_L)
    avg_mae_G = sum(mae_G) / len(mae_G)
    avg_mae_Q = sum(mae_Q) / len(mae_Q)
    avg_mae = (avg_mae_L + avg_mae_G + avg_mae_Q) / 3
    
    return {"L": avg_mae_L, "G": avg_mae_G, "Q": avg_mae_Q, "avg": avg_mae}

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
