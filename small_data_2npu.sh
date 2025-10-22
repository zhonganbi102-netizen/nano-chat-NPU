#!/bin/bash

echo "🧪 小数据2NPU测试..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:128"

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

echo "配置信息："
echo "  🎯 2个NPU (0,1)"
echo "  📊 超小模型：depth=2"
echo "  💾 最小batch：device_batch_size=1"
echo "  📏 短序列：max_seq_len=128"
echo "  🔢 极少步数：仅3步训练"
echo "  ⏭️  跳过所有评估"
echo ""

echo "启动小数据2NPU测试..."

torchrun --standalone --nproc_per_node=2 -- scripts/base_train.py \
    --depth=2 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --max_seq_len=128 \
    --num_iterations=3 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="tiny_2npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 小数据2NPU成功！可以尝试4NPU"
    echo "建议运行: ./small_data_4npu.sh"
else
    echo "❌ 小数据2NPU失败，需要进一步调试"
fi

echo ""
echo "小数据2NPU测试完成: $(date)"