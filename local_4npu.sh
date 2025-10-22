#!/bin/bash

echo "🎯 单机4NPU训练（本地通信优化）..."

# 基础环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0

# 使用本地通信，避免网络问题
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# HCCL优化设置（单机场景）
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_CONNECT_TIMEOUT=1800  # 30分钟超时
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=64

# 内存和性能优化
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"
export OMP_NUM_THREADS=1

# 清理之前的进程
echo "清理环境..."
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 5

# 清理NPU缓存
python3 -c "
import torch_npu
print('清理NPU缓存...')
for i in range(4):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    except:
        pass
print('缓存清理完成')
"

echo ""
echo "配置信息:"
echo "  NPU设备: 0,1,2,3"
echo "  模型深度: 6层"
echo "  单设备batch: 3"
echo "  总batch: 49152"
echo "  序列长度: 1024"
echo "  训练步数: 20"

echo ""
echo "启动单机4NPU训练..."

# 使用较小的配置，确保稳定性
torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    -- \
    scripts/base_train.py \
    --depth=6 \
    --device_batch_size=3 \
    --total_batch_size=49152 \
    --max_seq_len=1024 \
    --num_iterations=20 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="local_4npu_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "训练完成: $(date)"