#!/bin/bash

echo "🚀 无需wandb的4NPU训练..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_CONNECT_TIMEOUT=600

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"

# 禁用wandb
export WANDB_MODE=disabled

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 3

echo "启动无wandb的4NPU训练..."
echo "配置: 禁用wandb，避免登录问题"

torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=6 \
    --device_batch_size=3 \
    --total_batch_size=49152 \
    --max_seq_len=1024 \
    --num_iterations=20 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="dummy"

echo "训练完成: $(date)"