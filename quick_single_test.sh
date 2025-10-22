#!/bin/bash

echo "🔥 快速单NPU训练测试..."

# 清理进程
pkill -f "base_train" || true
sleep 1

# 设置环境
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

echo "📊 NPU设备状态:"
npu-smi info | head -10

echo "🚀 启动单NPU训练测试..."

python scripts/base_train.py \
    --device_batch_size=4 \
    --total_batch_size=32 \
    --max_seq_len=512 \
    --model_size=124M \
    --learning_rate=0.0006 \
    --warmup_iters=5 \
    --max_iters=20 \
    --eval_every=10 \
    --eval_tokens=2048 \
    --save_every=100 \
    --generate_every=100 \
    --overwrite_output_dir=True \
    --optimizer=adamw \
    --output_dir=./logs/quick_single_test

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ 单NPU快速测试成功！"
else
    echo "❌ 单NPU快速测试失败，退出码: $exit_code"
fi

echo "🏁 快速测试完成"