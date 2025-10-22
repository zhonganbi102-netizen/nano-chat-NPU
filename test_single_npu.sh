#!/bin/bash

echo "🧪 单NPU对比测试..."

# 使用单个NPU测试相同配置
export ASCEND_RT_VISIBLE_DEVICES=0

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
sleep 2

echo "启动单NPU测试（相同配置）..."
echo "配置: depth=4, batch=1, 5步训练"

python3 scripts/base_train.py \
    --depth=4 \
    --device_batch_size=1 \
    --total_batch_size=8192 \
    --max_seq_len=512 \
    --num_iterations=5 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="single_npu_test_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo "✅ 单NPU测试成功"
    echo "问题可能在分布式通信，而非基础训练"
else
    echo "❌ 单NPU测试失败"
    echo "问题在基础训练配置"
fi

echo "单NPU测试完成: $(date)"