#!/bin/bash

echo "🚀 快速4NPU分布式测试..."

# 清理进程
pkill -f "base_train" || true
sleep 2

# 设置环境
export HCCL_CONNECT_TIMEOUT=300
export HCCL_EXEC_TIMEOUT=300
export HCCL_HEARTBEAT_TIMEOUT=300
export HCCL_REDUCE_OP_SYNC=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12361
export PYTHONFAULTHANDLER=1

echo "🎯 启动4NPU快速测试..."

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/base_train.py \
    --device_batch_size=2 \
    --total_batch_size=4096 \
    --max_seq_len=512 \
    --depth=12 \
    --num_iterations=10 \
    --eval_every=10 \
    --eval_tokens=2048 \
    --core_metric_every=999999

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "🎉 4NPU快速测试成功！你的目标实现了！"
    echo "💪 现在你可以在4台NPU上训练了！"
else
    echo "❌ 4NPU快速测试失败，退出码: $exit_code"
fi

echo "🏁 4NPU测试完成"