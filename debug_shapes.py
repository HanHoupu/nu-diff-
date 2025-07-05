#!/usr/bin/env python3
"""
调试脚本：检查数据形状和模型输入
"""
import pathlib, sys
import torch
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.nucdiff.data.dataloader import get_loaders, build_dataset
from src.nucdiff.model import TransformerModel

def debug_data_shapes():
    """检查数据形状"""
    print("=== 数据形状调试 ===")
    
    # 加载配置
    with open("configs/default_clean.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 构建数据集
    train_ds, val_ds, (elem2idx, rec2idx, numeric_dim) = build_dataset(2004, cfg)
    print(f"数据集大小: train={len(train_ds)}, val={len(val_ds)}")
    print(f"映射字典: elem2idx={len(elem2idx)}, rec2idx={len(rec2idx)}")
    print(f"数值特征维度: {numeric_dim}")
    
    # 检查单个样本
    sample_x, sample_y = train_ds[0]
    print(f"\n单个样本形状:")
    print(f"  numeric: {sample_x['numeric'].shape}")
    print(f"  element: {sample_x['element'].shape}")
    print(f"  record_type: {sample_x['record_type'].shape}")
    print(f"  feature_type_ids: {sample_x['feature_type_ids'].shape}")
    print(f"  标签: L={sample_y['L'].shape}, G={sample_y['G'].shape}, Q={sample_y['Q'].shape}")
    
    # 检查批次数据
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    batch_x, batch_y = next(iter(train_loader))
    print(f"\n批次数据形状:")
    print(f"  numeric: {batch_x['numeric'].shape}")
    print(f"  element: {batch_x['element'].shape}")
    print(f"  record_type: {batch_x['record_type'].shape}")
    print(f"  feature_type_ids: {batch_x['feature_type_ids'].shape}")
    print(f"  标签: L={batch_y['L'].shape}, G={batch_y['G'].shape}, Q={batch_y['Q'].shape}")
    
    return batch_x, batch_y, elem2idx, rec2idx, numeric_dim

def debug_model_forward(batch_x, elem2idx, rec2idx, numeric_dim):
    """调试模型前向传播"""
    print("\n=== 模型前向传播调试 ===")
    
    # 加载配置
    with open("configs/default_clean.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 创建模型
    model = TransformerModel(cfg, elem2idx, rec2idx)
    print(f"模型创建成功")
    
    # 检查嵌入层
    print(f"嵌入层:")
    print(f"  element embedding: {model.elem_embed.num_embeddings} -> {model.elem_embed.embedding_dim}")
    print(f"  record embedding: {model.rec_embed.num_embeddings} -> {model.rec_embed.embedding_dim}")
    print(f"  feature_type embedding: {model.feature_type_embed.num_embeddings} -> {model.feature_type_embed.embedding_dim}")
    print(f"  numeric projection: {model.numeric_proj.in_features} -> {model.numeric_proj.out_features}")
    
    # 手动检查嵌入过程
    print(f"\n手动检查嵌入过程:")
    
    # 1. 数值特征嵌入
    numeric = batch_x['numeric'].unsqueeze(-1)
    print(f"  numeric after unsqueeze: {numeric.shape}")
    numeric_emb = model.numeric_proj(numeric)
    print(f"  numeric embedding: {numeric_emb.shape}")
    
    # 2. 特征类型嵌入
    feature_type_emb = model.feature_type_embed(batch_x['feature_type_ids'])
    print(f"  feature_type embedding: {feature_type_emb.shape}")
    
    # 3. 元素嵌入
    elem_emb = model.elem_embed(batch_x['element'])
    print(f"  element embedding: {elem_emb.shape}")
    
    # 4. 记录类型嵌入
    rec_emb = model.rec_embed(batch_x['record_type'])
    print(f"  record embedding: {rec_emb.shape}")
    
    # 5. 检查是否可以相加
    print(f"\n检查张量相加:")
    print(f"  numeric_emb: {numeric_emb.shape}")
    print(f"  feature_type_emb: {feature_type_emb.shape}")
    print(f"  elem_emb: {elem_emb.shape}")
    print(f"  rec_emb: {rec_emb.shape}")
    
    try:
        # 扩展分类特征到序列长度
        batch_size = batch_x['element'].shape[0]
        seq_len = batch_x['numeric'].shape[1]
        
        # feature_type_emb已经是 [B, 15, 128]，不需要扩展
        # elem_emb和rec_emb需要从 [B, 128] 扩展到 [B, 15, 128]
        elem_emb_expanded = elem_emb.unsqueeze(1).expand(-1, seq_len, -1)
        rec_emb_expanded = rec_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = numeric_emb + feature_type_emb + elem_emb_expanded + rec_emb_expanded
        print(f"  成功相加，结果形状: {combined.shape}")
    except RuntimeError as e:
        print(f"  相加失败: {e}")
        return False
    
    return True

def main():
    """主函数"""
    try:
        # 调试数据形状
        batch_x, batch_y, elem2idx, rec2idx, numeric_dim = debug_data_shapes()
        
        # 调试模型前向传播
        success = debug_model_forward(batch_x, elem2idx, rec2idx, numeric_dim)
        
        if success:
            print("\n✅ 调试完成，模型前向传播正常")
        else:
            print("\n❌ 发现问题，需要修复")
            
    except Exception as e:
        print(f"\n❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 