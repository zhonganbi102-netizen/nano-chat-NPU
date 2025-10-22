#!/bin/bash

echo "🔥 快速2NPU分布式测试..."

# 清理进程
pkill -f "base_train" || true
sleep 2

# 设置环境
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=120
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12360
export PYTHONFAULTHANDLER=1

echo "🚀 启动2NPU测试..."

torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/base_train.py \
    --device_batch_size=2 \
    --total_batch_size=2048 \
    --max_seq_len=512 \
    --depth=12 \
    --num_iterations=10 \
    --eval_every=10 \
    --eval_tokens=2048 \
    --core_metric_every=999999

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ 2NPU快速测试成功！"
else
    echo "❌ 2NPU快速测试失败，退出码: $exit_code"
fi

echo "🏁 2NPU测试完成"