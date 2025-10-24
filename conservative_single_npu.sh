#!/bin/bash

# 保守单NPU训练 - 解决Muon优化器卡死问题
# Conservative single NPU training - fix Muon optimizer hanging

set -e

echo "🛡️  保守单NPU训练 - 解决Muon优化器问题"
echo ""

# 1. 环境设置
echo "1. 设置NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 严格的单NPU设置
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29700

# 内存和编译优化
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64  # NPU最小要求>20MB
export NPU_COMPILE_DISABLE=1
export TORCH_NPU_DISABLE_LAZY_INIT=1

# Python优化
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "✅ 环境配置完成"

# 2. 彻底清理
echo ""
echo "2. 彻底清理环境..."
pkill -f "python.*train" || true
pkill -f "torchrun" || true
sleep 2

python3 -c "
import torch
import torch_npu
import gc
import time

print('清理NPU内存...')
if torch_npu.npu.is_available():
    for i in range(torch_npu.npu.device_count()):
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    
    # 等待清理完成
    time.sleep(1)
    gc.collect()
    time.sleep(1)
    
    print(f'✅ 已清理 {torch_npu.npu.device_count()} 个NPU设备')
"

# 3. 检查基础环境
echo ""
echo "3. 检查基础环境..."
python3 -c "
import torch
import torch_npu
import sys

print(f'Python版本: {sys.version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'torch_npu版本: {torch_npu.__version__}')
print(f'NPU可用: {torch_npu.npu.is_available()}')
print(f'NPU设备数: {torch_npu.npu.device_count()}')

if torch_npu.npu.is_available():
    torch_npu.npu.set_device(0)
    print(f'当前设备: npu:{torch_npu.npu.current_device()}')
    
    # 测试简单张量操作
    x = torch.randn(2, 2).to('npu:0')
    y = x + 1
    print(f'张量测试: {y.sum().item():.2f}')
    print('✅ NPU基础功能正常')
"

# 4. 保守训练参数
echo ""
echo "4. 开始保守训练..."
echo "使用最小可行参数避免Muon卡死..."

# 创建训练启动脚本
cat > temp_conservative_train.py << 'EOF'
import os
import sys
import time
import torch
import torch_npu

# 设置NPU设备
if torch_npu.npu.is_available():
    torch_npu.npu.set_device(0)
    print(f"使用设备: npu:{torch_npu.npu.current_device()}")

# 导入训练脚本
sys.path.append('.')
from scripts.base_train import main

if __name__ == "__main__":
    print("🚀 开始保守单NPU训练...")
    
    # 设置最保守的参数
    sys.argv = [
        'base_train.py',
        '--run=conservative_single_npu',
        '--depth=4',                    # 极小深度
        '--device_batch_size=1',        # 最小batch
        '--total_batch_size=2',         # 最小总batch
        '--num_iterations=20',          # 少量迭代
        '--embedding_lr=0.0001',        # 极小学习率
        '--unembedding_lr=0.00001',
        '--matrix_lr=0.00005',
        '--grad_clip=1.0',
        '--eval_every=10',
        '--sample_every=999999',
        '--core_metric_every=999999',
        '--verbose'
    ]
    
    try:
        main()
        print("✅ 保守训练成功完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

# 执行保守训练
python3 temp_conservative_train.py

# 清理临时文件
rm -f temp_conservative_train.py

echo ""
echo "🎉 保守单NPU训练完成！"
echo ""
echo "如果成功，可以尝试增加参数："
echo "  - depth: 4 -> 6 -> 8"
echo "  - device_batch_size: 1 -> 2 -> 4"
echo "  - num_iterations: 20 -> 100 -> 1000"