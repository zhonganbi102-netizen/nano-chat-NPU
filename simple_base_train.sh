#!/bin/bash

# 简化版本 - 只运行base model训练
# Simplified version - only base model training

set -e
echo "=== 简化版Base Model训练 ==="

# NPU环境检查
python3 -c "
import torch
import torch_npu
assert torch_npu.npu.is_available(), 'NPU不可用'
print(f'NPU设备数量: {torch_npu.npu.device_count()}')
"

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "开始4-NPU base model训练（正确配置）..."
echo "配置说明："
echo "  device_batch_size: 256"
echo "  max_seq_len: 2048 (默认)"  
echo "  world_size: 4"
echo "  world_tokens_per_fwdbwd: 256 * 2048 * 4 = 2,097,152"
echo "  total_batch_size: 2,097,152 (确保整除)"
echo "  grad_accum_steps: 1"

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --run=npu_base_d12 \
    --depth=12 \
    --device_batch_size=256 \
    --total_batch_size=2097152

echo "训练完成！"