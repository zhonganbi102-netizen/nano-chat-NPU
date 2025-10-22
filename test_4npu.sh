#!/bin/bash

echo "🧪 4NPU快速测试..."

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

echo "启动4NPU测试训练 (100步)..."

# 最小化配置测试
torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --run "4npu_test_$(date +%Y%m%d_%H%M%S)" \
    --num_iterations 100

echo "测试完成: $(date)"