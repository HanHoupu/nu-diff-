# Nuclear Data Prediction - Enhanced Training

## 概述

本项目实现了核数据预测的增强训练系统，支持FP16、平衡损失函数、分层抽样等优化功能。

## 主要特性

### 🚀 性能优化
- **FP16训练**: 减少显存占用，提升训练速度
- **大批量训练**: 支持batch_size=2048
- **优化数据加载**: 多进程、预取、持久化工作进程
- **梯度裁剪**: 稳定大学习率训练

### 📊 数据增强
- **物理差分特征**: 添加ΔE、A_ratio、ΔA、ΔZ等特征
- **分层抽样**: 按核素和记录类型分层，防止数据泄漏
- **平衡损失函数**: 解决多任务量纲差异问题

### 🎯 模型优化
- **Mid-Plus架构**: d_model=256, n_layers=4, 约14M参数
- **LoRA支持**: 为后续微调预留接口
- **多任务学习**: L/G/Q三个任务联合训练

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 检查GPU
nvidia-smi
```

### 2. 数据准备

确保`clean data/`目录下有ENSDF数据文件：
- `ENSDF-2004.L`
- `ENSDF-2004.G` 
- `ENSDF-2004.Q`

### 3. 训练命令

```bash
# 使用增强训练脚本
python -m src.nucdiff.train_enhanced --year 2004 --cfg configs/mid_plus.yaml --loss balanced

# 使用原始训练脚本（对比）
python -m src.nucdiff.train --year 2004 --cfg configs/default.yaml
```

### 4. 监控训练

```bash
# 实时监控显存
watch -n1 nvidia-smi

# 查看TensorBoard日志
tensorboard --logdir=runs --port=6006
```

## 配置说明

### Mid-Plus配置 (`configs/mid_plus.yaml`)

```yaml
# 模型配置
d_model: 256
n_layers: 4
n_heads: 8
ff_dim: 1024

# 训练配置
batch_size: 2048
lr: 1e-3
max_epochs: 60
early_stop_patience: 10

# 性能优化
amp: true  # FP16
grad_clip_norm: 1.0
num_workers: 8
pin_memory: true
```

### 损失函数选项

- `--loss balanced`: 使用平衡损失函数（推荐）
- `--loss simple`: 使用简单损失函数

## 预期性能

### 20 epoch后期望指标
- **MAE_L**: ≈ 1 keV
- **MAE_G**: ≤ 2 keV  
- **MAE_Q**: ≤ 120 keV
- **R²_L**: ≥ 0.8
- **R²_G**: ≥ 0.6
- **R²_Q**: ≥ 0.4

### 显存使用
- **FP16 + batch 2048**: ≈ 8 GB
- **3090 Ti**: 剩余6 GB可用于梯度累积

## 文件结构

```
src/nucdiff/
├── data/
│   ├── dataloader.py          # 原始数据加载器
│   └── enhanced_dataloader.py # 增强数据加载器
├── model/
│   └── transformer_model.py   # Transformer模型
├── utils/
│   ├── balanced_loss.py       # 平衡损失函数
│   ├── evaluate.py            # 评估函数
│   └── fisher.py             # 正则化
├── train.py                   # 原始训练脚本
└── train_enhanced.py         # 增强训练脚本
```

## 测试

```bash
# 运行增强功能测试
python test_enhanced.py

# 运行原始功能测试
python test_fixes.py
```

## 故障排除

### 常见问题

1. **显存不足**
   - 降低batch_size到1024
   - 关闭FP16: `amp: false`

2. **数据加载慢**
   - 减少num_workers到4
   - 关闭persistent_workers

3. **训练不稳定**
   - 降低学习率到5e-4
   - 增加grad_clip_norm到2.0

### 监控命令

```bash
# 显存监控
watch -n1 nvidia-smi

# 进程监控
htop

# 日志查看
tail -f outputs/run-*/logs/training.log
```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License 