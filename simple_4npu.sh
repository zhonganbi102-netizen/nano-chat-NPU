#!/bin/bash

echo "⚡ 简化4NPU训练（无评估）..."

# 基础环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_CONNECT_TIMEOUT=300

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 3

echo "启动简化4NPU训练..."
echo "配置: 无初始评估，专注训练循环"

# 直接开始训练，禁用所有评估
torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=6 \
    --device_batch_size=4 \
    --total_batch_size=65536 \
    --max_seq_len=1024 \
    --num_iterations=20 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="4npu_simple_$(date +%Y%m%d_%H%M%S)"

echo "简化训练完成: $(date)"