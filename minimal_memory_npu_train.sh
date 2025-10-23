#!/bin/bash

# 超低内存使用的NPU训练脚本
# 专门针对内存严重不足的情况

set -e

echo "=== 超低内存NPU训练 ==="

# 清理环境
./emergency_npu_cleanup.sh

# 等待清理完成
sleep 10

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0  # 只使用一个NPU
export WORLD_SIZE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64  # 限制内存分片

echo "开始超低内存训练..."

# 使用最小的batch size
python -m scripts.base_train \
    --run=minimal_npu \
    --depth=6 \
    --device_batch_size=8 \
    --total_batch_size=8192 \
    --max_steps=1000

echo "训练完成！"