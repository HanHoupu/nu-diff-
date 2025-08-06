#!/usr/bin/env python3
"""
渐进式overfit测试脚本
测试模型在不同样本数量下的记忆能力
"""

import subprocess
import sys

def run_overfit_test(samples, seed=412, max_epochs=100):
    """运行overfit测试"""
    cmd = [
        sys.executable, "-m", "src.nucdiff.train_slim",
        "--cfg", "configs/temporal.yaml", 
        "--task", "SEQ",
        "--overfit",
        "--overfit_samples", str(samples),
        "--train_head", "classification"
    ]
    
    print(f"\n🧪 Testing overfit with {samples} samples (seed={seed})")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # 运行训练
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def main():
    """渐进式测试序列"""
    test_sequence = [5, 10, 20, 50]  # 逐步增加样本数
    
    print("🎯 渐进式Overfit测试序列")
    print("目标：观察模型在不同样本数下的记忆能力")
    
    for samples in test_sequence:
        success = run_overfit_test(samples)
        if not success:
            print(f"❌ {samples}样本测试失败，停止后续测试")
            break
        print(f"✅ {samples}样本测试完成")
        
        # 询问是否继续
        if samples < max(test_sequence):
            response = input(f"\n继续测试{test_sequence[test_sequence.index(samples)+1]}样本? (y/n): ")
            if response.lower() != 'y':
                break

if __name__ == "__main__":
    main()