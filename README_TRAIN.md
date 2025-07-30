# 统一训练脚本使用说明

## 🎯 **功能概述**

统一的`train.py`脚本支持多种训练模式，满足不同的训练需求。

## 📋 **训练模式**

### 1. **时间分段训练** (Temporal Training)
```bash
# 使用配置文件中的时间范围
python -m nucdiff.train --cfg configs/temporal.yaml --task L --mode temporal

# 自定义时间范围
python -m nucdiff.train --cfg configs/temporal.yaml --task L --mode temporal \
  --start_year 2004 --end_year 2010 \
  --val_start_year 2010 --val_end_year 2012
```

### 2. **单任务训练** (Single Task)
```bash
# 使用配置文件中的年份
python -m nucdiff.train --cfg configs/single.yaml --task L --mode single

# 自定义年份
python -m nucdiff.train --cfg configs/single.yaml --task L --mode single --year 2004
```

### 3. **LoRA微调** (LoRA Fine-tuning)
```bash
# 基于预训练模型进行LoRA微调
python -m nucdiff.train --cfg configs/lora.yaml --mode lora \
  --base_model runs/L_2004_xxx/checkpoints/model_L_2004.pth
```

### 4. **测试模式** (Test Mode)
```bash
# 只加载数据不训练，用于验证数据加载
python -m nucdiff.train --cfg configs/temporal.yaml --task L --test
```

## 📁 **配置文件**

### `configs/temporal.yaml` - 时间分段训练
- 支持时间范围配置
- 专注单任务训练
- 适合时间连续性验证

### `configs/single.yaml` - 单任务训练
- 单年数据训练
- 专注特定任务
- 适合快速实验

### `configs/lora.yaml` - LoRA微调
- 基于预训练模型
- 只训练LoRA参数
- 适合增量学习

## 🎛️ **命令行参数**

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--cfg` | 配置文件路径 | ✅ | - |
| `--task` | 任务类型 (L/G/Q) | ❌ | - |
| `--mode` | 训练模式 | ❌ | temporal |
| `--test` | 测试模式 | ❌ | False |
| `--start_year` | 开始年份 | ❌ | 配置文件 |
| `--end_year` | 结束年份 | ❌ | 配置文件 |
| `--val_start_year` | 验证开始年份 | ❌ | 配置文件 |
| `--val_end_year` | 验证结束年份 | ❌ | 配置文件 |
| `--year` | 训练年份 | ❌ | 配置文件 |
| `--base_model` | 预训练模型路径 | ❌ | - |

## 📊 **输出结构**

```
runs/
├── temporal_L_2004-2010_20250101_120000/
│   ├── checkpoints/
│   │   ├── model_temporal_L_2004-2010.pth
│   │   └── lora_temporal_L_2004-2010.pth
│   ├── logs/
│   ├── config.yaml
│   ├── cmd.txt
│   └── requirements.txt
└── ...
```

## 🚀 **使用示例**

### 时间分段训练L任务
```bash
python -m nucdiff.train --cfg configs/temporal.yaml --task L --mode temporal
```

### 单任务训练G任务
```bash
python -m nucdiff.train --cfg configs/single.yaml --task G --mode single --year 2005
```

### LoRA微调
```bash
python -m nucdiff.train --cfg configs/lora.yaml --mode lora \
  --base_model runs/L_2004_xxx/checkpoints/model_L_2004.pth
```

### 测试数据加载
```bash
python -m nucdiff.train --cfg configs/temporal.yaml --task L --test
```

## ⚠️ **注意事项**

1. **时间分段模式**：需要确保数据文件存在
2. **LoRA微调**：需要先有预训练模型
3. **测试模式**：只验证数据加载，不进行训练
4. **配置文件**：根据实际需求调整参数

## 🔧 **故障排除**

### 常见问题

1. **数据文件不存在**
   - 检查`clean data/`目录
   - 确保ENSDF文件存在

2. **显存不足**
   - 降低`batch_size`
   - 减少模型参数

3. **训练不稳定**
   - 降低学习率
   - 增加`grad_clip_norm`

4. **早停过早**
   - 增加`early_stop_patience`
   - 调整`min_delta` 