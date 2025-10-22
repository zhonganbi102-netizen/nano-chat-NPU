#!/bin/bash

echo "🔥 极简单NPU测试..."

# 清理进程
pkill -f "base_train" || true
sleep 1

echo "🚀 启动极简训练..."

python scripts/base_train.py \
    --device_batch_size=2 \
    --total_batch_size=512 \
    --max_seq_len=256 \
    --depth=6 \
    --num_iterations=5 \
    --eval_every=5 \
    --core_metric_every=999999

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ 极简NPU测试成功！"
else
    echo "❌ 极简NPU测试失败，退出码: $exit_code"
fi

echo "🏁 极简测试完成"