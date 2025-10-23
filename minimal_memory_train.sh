#!/bin/bash

# 极小batch size版本 - 确保不会OOM
# Minimal batch size version - ensure no OOM

set -e
echo "=== 极小配置Base Model训练 ==="

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

echo "开始4-NPU base model训练（极小配置）..."
echo "配置说明："
echo "  device_batch_size: 64 (极小batch避免OOM)"
echo "  max_seq_len: 2048 (默认)"  
echo "  world_size: 4"
echo "  world_tokens_per_fwdbwd: 64 * 2048 * 4 = 524,288"
echo "  total_batch_size: 524,288 (确保整除)"
echo "  grad_accum_steps: 1"

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --run=npu_base_d12_minimal \
    --depth=12 \
    --device_batch_size=64 \
    --total_batch_size=524288

echo "训练完成！"