#!/bin/bash

# 修复权限并运行FineWeb训练
# Fix permissions and run FineWeb training

set -e

echo "=== 修复权限并启动FineWeb训练 ==="

# 1. 修复所有脚本的执行权限
echo "1. 修复执行权限..."
chmod +x *.sh 2>/dev/null || true
chmod +x *.py 2>/dev/null || true

# 2. 强制清理NPU环境
echo "2. 强制清理NPU环境..."
pkill -f python || echo "没有Python进程"
sleep 3

# 清理NPU缓存
python3 -c "
import torch
try:
    import torch_npu
    import gc
    if torch_npu.npu.is_available():
        for i in range(torch_npu.npu.device_count()):
            torch_npu.npu.set_device(i)
            torch_npu.npu.empty_cache()
        gc.collect()
        print('✅ NPU缓存已清理')
except Exception as e:
    print(f'清理失败: {e}')
" || echo "NPU清理失败，继续..."

# 3. 设置优化的环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# NPU优化设置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export NPU_COMPILE_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "环境配置:"
echo "  PYTORCH_NPU_ALLOC_CONF: $PYTORCH_NPU_ALLOC_CONF"
echo "  编译优化: 禁用"

# 4. 检查数据文件
echo "3. 检查FineWeb数据..."
if [ -d "~/.cache/nanochat/tokenized_data" ]; then
    data_count=$(find ~/.cache/nanochat/tokenized_data -name "*.parquet" | wc -l)
    echo "✅ 发现 $data_count 个数据文件"
else
    echo "❌ 数据目录不存在，需要先下载数据"
    exit 1
fi

# 5. 训练tokenizer（如果需要）
echo "4. 检查tokenizer..."
if [ ! -f "~/.cache/nanochat/tokenizer/tokenizer.pkl" ]; then
    echo "训练tokenizer..."
    python -m scripts.tok_train || echo "tokenizer训练失败，继续..."
else
    echo "✅ tokenizer已存在"
fi

# 6. 启动NPU训练
echo "5. 启动FineWeb NPU训练..."
echo "配置: depth=8, batch_size=16, total_batch_size=32768"

python -c "
import os
import sys
import torch
import torch_npu

# 禁用编译
torch._dynamo.config.disable = True
torch.set_default_device('npu:0')

# 添加路径
sys.path.append('.')

# 设置训练参数 - 适合FineWeb大规模训练
sys.argv = [
    'base_train.py',
    '--run=fineweb_npu_training',
    '--depth=8',
    '--device_batch_size=16',
    '--total_batch_size=32768',
    '--num_iterations=10000'  # 大规模训练
]

# 运行训练
from scripts import base_train
print('🚀 开始FineWeb大规模NPU训练...')
base_train.main()
"

echo "✅ FineWeb训练完成！"