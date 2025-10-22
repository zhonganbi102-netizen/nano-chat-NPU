#!/bin/bash

echo "🔧 使用修复优化器的2NPU测试..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb=128"

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

echo "配置信息："
echo "  🔧 已修复优化器：使用标准AdamW和Muon"
echo "  🎯 2个NPU (0,1)"
echo "  📊 小模型：depth=3"
echo "  💾 小batch：device_batch_size=1"
echo "  📏 短序列：max_seq_len=256"
echo "  🔢 少步数：仅5步训练"
echo ""

echo "启动修复优化器的2NPU测试..."

torchrun --standalone --nproc_per_node=2 -- scripts/base_train.py \
    --depth=3 \
    --device_batch_size=1 \
    --total_batch_size=2048 \
    --max_seq_len=256 \
    --num_iterations=5 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="fixed_opt_2npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 修复优化器2NPU测试成功！"
    echo "现在可以尝试4NPU: ./fixed_opt_4npu.sh"
else
    echo ""
    echo "❌ 修复优化器2NPU测试失败"
    echo "可能还有其他问题"
fi

echo ""
echo "修复优化器测试完成: $(date)"