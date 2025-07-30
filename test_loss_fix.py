#!/usr/bin/env python3

import torch
import numpy as np
from src.nucdiff.data.dataloader import build_unified_dataset
from src.nucdiff.model.transformer_model import TransformerModel
# 不再需要导入自定义loss函数
import yaml

print("=== 测试双重标准化修复 ===")

# 加载配置
with open('configs/single.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# 临时禁用LoRA以避免兼容性问题
cfg['use_lora'] = False

# 构建数据集
train_ds, val_ds, metadata = build_unified_dataset(2004, 2004, "L", cfg, debug=False)

# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(cfg, metadata['elem2idx'], metadata['rec2idx']).to(device)

# 获取一个批次
from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
batch_x, batch_y = next(iter(train_loader))

# 移动到设备
for k, v in batch_x.items():
    batch_x[k] = v.to(device)
for k, v in batch_y.items():
    batch_y[k] = v.to(device)

print(f"标签统计 (应该是均值~0, 方差~1):")
y_L = batch_y['L']
valid_mask = torch.isfinite(y_L)
y_valid = y_L[valid_mask]
print(f"  - 有效标签数量: {valid_mask.sum()}")
print(f"  - 标签均值: {y_valid.mean():.4f}")
print(f"  - 标签方差: {y_valid.var():.4f}")
print(f"  - 标签范围: [{y_valid.min():.4f}, {y_valid.max():.4f}]")

# 测试常数预测的loss
print(f"\n=== 测试常数预测 ===")
mse_loss = torch.nn.MSELoss()

# 模拟预测常数0 (标准化后的均值)
pred_zero = torch.zeros_like(y_L)
valid_mask = torch.isfinite(y_L)
if valid_mask.sum() > 0:
    loss_zero = mse_loss(pred_zero[valid_mask], y_L[valid_mask])
    loss_zero_str = f"{loss_zero:.4f}"
else:
    loss_zero_str = "None"
print(f"预测常数0的loss: {loss_zero_str} (应该约等于标签方差 {y_valid.var():.4f})")

# 模拟预测标签均值
pred_mean = torch.full_like(y_L, y_valid.mean())
if valid_mask.sum() > 0:
    loss_mean = mse_loss(pred_mean[valid_mask], y_L[valid_mask])
    loss_mean_str = f"{loss_mean:.4f}"
else:
    loss_mean_str = "None"
print(f"预测标签均值的loss: {loss_mean_str}")

# 测试实际模型预测
print(f"\n=== 测试实际模型预测 ===")
model.eval()
with torch.no_grad():
    predictions = model(batch_x)

pred_L = predictions['L']
pred_valid = pred_L[valid_mask]
print(f"模型预测统计:")
print(f"  - 预测均值: {pred_valid.mean():.4f}")
print(f"  - 预测方差: {pred_valid.var():.4f}")
print(f"  - 预测范围: [{pred_valid.min():.4f}, {pred_valid.max():.4f}]")

# 检查是否输出常数
pred_std = pred_valid.std()
print(f"  - 预测标准差: {pred_std:.6f}")
if pred_std < 1e-3:
    print("  - ⚠️  模型可能在输出常数!")
else:
    print("  - ✅ 模型输出有变化")

# 计算实际loss
pred_L = predictions['L']
if valid_mask.sum() > 0:
    actual_loss = mse_loss(pred_L[valid_mask], y_L[valid_mask])
    actual_loss_str = f"{actual_loss:.4f}"
else:
    actual_loss = None
    actual_loss_str = "None"
print(f"实际loss: {actual_loss_str}")

print(f"\n=== 结论 ===")
print(f"如果修复成功，实际loss应该小于常数预测的loss ({loss_zero_str})")
print(f"实际loss: {actual_loss_str}")
if actual_loss is not None and loss_zero is not None and actual_loss < loss_zero:
    print("✅ 修复成功！模型学到了比常数预测更好的策略")
else:
    print("❌ 可能还有问题，模型仍在输出接近常数的预测") 