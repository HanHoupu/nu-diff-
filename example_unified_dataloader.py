#!/usr/bin/env python3
"""
整合后的数据集构建功能使用示例
"""

import sys
sys.path.append('src')

from nucdiff.data.dataloader import build_unified_dataset, load_records_for_task

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 配置
    cfg = {
        "train_frac": 0.8,
        "seed": 42
    }
    
    # 1. 单年L任务数据集
    print("\n1. 构建单年L任务数据集...")
    train_ds, val_ds, metadata = build_unified_dataset(
        start_year=2004,
        end_year=2004,
        task="L",
        cfg=cfg,
        debug=True
    )
    print(f"✅ 成功构建数据集!")
    print(f"  - 训练集: {len(train_ds)} 样本")
    print(f"  - 验证集: {len(val_ds)} 样本")
    print(f"  - 元素映射: {len(metadata['elem2idx'])} 个元素")
    
    # 2. 多年L任务数据集
    print("\n2. 构建多年L任务数据集...")
    train_ds, val_ds, metadata = build_unified_dataset(
        start_year=2004,
        end_year=2006,
        task="L",
        cfg=cfg,
        debug=False
    )
    print(f"✅ 成功构建多年数据集!")
    print(f"  - 训练集: {len(train_ds)} 样本")
    print(f"  - 验证集: {len(val_ds)} 样本")
    
    # 3. 不同任务类型
    print("\n3. 构建不同任务类型数据集...")
    for task in ["L", "G", "Q"]:
        try:
            train_ds, val_ds, metadata = build_unified_dataset(
                start_year=2004,
                end_year=2004,
                task=task,
                cfg=cfg,
                debug=False
            )
            print(f"✅ {task}任务: 训练{len(train_ds)} | 验证{len(val_ds)}")
        except Exception as e:
            print(f"❌ {task}任务失败: {e}")

def example_debug_mode():
    """调试模式示例"""
    print("\n=== 调试模式示例 ===")
    
    cfg = {"train_frac": 0.8, "seed": 42}
    
    # 启用调试模式，查看详细的数据加载过程
    train_ds, val_ds, metadata = build_unified_dataset(
        start_year=2004,
        end_year=2004,
        task="L",
        cfg=cfg,
        debug=True  # 启用调试模式
    )
    
    print(f"✅ 调试模式完成!")

def example_data_loading():
    """数据加载示例"""
    print("\n=== 数据加载示例 ===")
    
    # 直接加载原始数据
    df = load_records_for_task(2004, 2004, "L", debug=True)
    print(f"✅ 直接加载数据: {len(df)} 条记录")
    print(f"  - 列名: {list(df.columns)}")
    if "energy_keV" in df.columns:
        print(f"  - energy_keV有效数: {df['energy_keV'].notna().sum()}")

if __name__ == "__main__":
    print("整合后的数据集构建功能使用示例")
    print("=" * 50)
    
    # 运行示例
    example_basic_usage()
    example_debug_mode()
    example_data_loading()
    
    print("\n🎉 所有示例运行完成!")
    print("\n使用方法:")
    print("1. 基本使用: build_unified_dataset(start_year, end_year, task, cfg, debug=False)")
    print("2. 调试模式: build_unified_dataset(..., debug=True)")
    print("3. 直接加载: load_records_for_task(start_year, end_year, task, debug=False)") 