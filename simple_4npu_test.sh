#!/bin/bash

echo "🧪 超简单4NPU测试..."

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 清理进程
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

echo "启动4NPU测试 (仅10步)..."

# 只传递必要参数
torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --num_iterations=10

echo "测试完成: $(date)"