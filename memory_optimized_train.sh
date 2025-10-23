#!/bin/bash

# 内存优化版本 - 减少batch size避免OOM
# Memory optimized version - reduce batch size to avoid OOM

set -e
echo "=== 内存优化版Base Model训练 ==="

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

echo "开始4-NPU base model训练（内存优化配置）..."
echo "配置说明："
echo "  device_batch_size: 128 (减少到128避免OOM)"
echo "  max_seq_len: 2048 (默认)"  
echo "  world_size: 4"
echo "  world_tokens_per_fwdbwd: 128 * 2048 * 4 = 1,048,576"
echo "  total_batch_size: 1,048,576 (确保整除)"
echo "  grad_accum_steps: 1"

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --run=npu_base_d12_mem_opt \
    --depth=12 \
    --device_batch_size=128 \
    --total_batch_size=1048576

echo "训练完成！"