#!/bin/bash

echo "🚀 使用修复优化器的4NPU测试..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
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
sleep 3

echo "配置信息："
echo "  🔧 已修复优化器：使用标准AdamW和Muon"
echo "  🎯 4个NPU (0,1,2,3)"
echo "  📊 小模型：depth=4"
echo "  💾 小batch：device_batch_size=2"
echo "  📏 中序列：max_seq_len=512"
echo "  🔢 训练步数：10步"
echo ""

echo "启动修复优化器的4NPU测试..."

torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=4 \
    --device_batch_size=2 \
    --total_batch_size=16384 \
    --max_seq_len=512 \
    --num_iterations=10 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="fixed_opt_4npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 修复优化器4NPU测试成功！"
    echo "多NPU分布式训练已完全正常！"
else
    echo ""
    echo "❌ 修复优化器4NPU测试失败"
fi

echo ""
echo "修复优化器4NPU测试完成: $(date)"