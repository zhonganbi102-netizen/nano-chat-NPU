#!/bin/bash

# 禁用编译优化的单NPU训练脚本
# No-compile single NPU training script

set -e

echo "=== 禁用编译优化的单NPU训练 ==="
echo "No-compile Single NPU Training"

# 1. 强制清理环境
echo "1. 强制清理NPU环境..."
pkill -f python || echo "没有Python进程"
sleep 5

# 重置NPU设备
python3 -c "
import torch
import torch_npu
import gc
if torch_npu.npu.is_available():
    for i in range(torch_npu.npu.device_count()):
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
    gc.collect()
    print('NPU缓存已清理')
" || echo "清理失败，继续..."

# 2. 设置禁用编译的环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 禁用编译优化的关键设置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export NPU_COMPILE_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "禁用编译配置:"
echo "  NPU_COMPILE_DISABLE: $NPU_COMPILE_DISABLE"
echo "  TORCH_COMPILE_DISABLE: $TORCH_COMPILE_DISABLE"
echo "  TORCHDYNAMO_DISABLE: $TORCHDYNAMO_DISABLE"

# 3. 检查NPU状态
echo "2. 检查NPU状态..."
npu-smi info || echo "npu-smi命令不可用"

# 4. 训练tokenizer（如果需要）
echo "3. 训练tokenizer（如果需要）..."
python -m scripts.tok_train || echo "tokenizer训练失败，继续..."

# 5. 禁用编译的单NPU训练
echo "4. 开始禁用编译的单NPU训练..."
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

# 设置训练参数
sys.argv = [
    'base_train.py',
    '--run=no_compile_single_npu',
    '--depth=6',
    '--device_batch_size=8',
    '--total_batch_size=16384',
    '--num_iterations=1000'
]

# 运行训练
from scripts import base_train
print('开始无编译训练...')
base_train.main()
"

echo "✅ 禁用编译训练完成！"