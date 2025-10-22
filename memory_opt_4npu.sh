#!/bin/bash

echo "💾 4NPU内存优化训练..."

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# NPU内存优化设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"

# 清理进程和内存
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 3

# 清理NPU内存
python3 -c "
import torch_npu
for i in range(4):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        print(f'清理NPU {i} 内存')
    except:
        pass
"

echo "启动内存优化的4NPU训练..."
echo "配置: 4个NPU, depth=8 (减小模型), device_batch_size=4 (减小batch), 梯度累积"

# 内存优化配置：更小的模型和batch size
torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=8 \
    --device_batch_size=4 \
    --total_batch_size=131072 \
    --max_seq_len=1024 \
    --num_iterations=100 \
    --run="4npu_memory_opt_$(date +%Y%m%d_%H%M%S)"

echo "训练完成: $(date)"