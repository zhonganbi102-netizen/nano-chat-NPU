#!/bin/bash

echo "=== 简化版单NPU训练 ==="

# 设置环境
export ASCEND_RT_VISIBLE_DEVICES=0
cd /mnt/linxid615/bza/nanochat-npu

echo "1. 停止现有进程..."
pkill -f python
sleep 5

echo "2. 开始简化训练..."

# 使用简化参数，绕过可能的问题
python3 -m scripts.base_train \
    --run=npu_simplified \
    --depth=2 \
    --device_batch_size=2 \
    --total_batch_size=4096 \
    --num_iterations=3 \
    --eval_every=2 \
    --sample_every=10 \
    --core_metric_every=10 \
    --matrix_lr=0.01 \
    --embedding_lr=0.1 \
    --unembedding_lr=0.002

echo "简化训练完成"