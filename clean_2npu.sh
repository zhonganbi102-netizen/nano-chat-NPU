#!/bin/bash

echo "🔧 修复NPU内存配置的2NPU测试..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# NPU内存设置（移除CUDA格式的参数）
unset PYTORCH_NPU_ALLOC_CONF

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

echo "配置信息："
echo "  🔧 已修复优化器：使用标准AdamW和Muon"
echo "  💾 移除不兼容的内存配置"
echo "  🎯 2个NPU (0,1)"
echo "  📊 小模型：depth=3"
echo ""

echo "启动NPU兼容的2NPU测试..."

torchrun --standalone --nproc_per_node=2 -- scripts/base_train.py \
    --depth=3 \
    --device_batch_size=1 \
    --total_batch_size=2048 \
    --max_seq_len=256 \
    --num_iterations=5 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="clean_2npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 NPU兼容2NPU测试成功！"
    echo "现在可以尝试4NPU: ./clean_4npu.sh"
else
    echo ""
    echo "❌ NPU兼容2NPU测试失败"
fi

echo ""
echo "NPU兼容测试完成: $(date)"