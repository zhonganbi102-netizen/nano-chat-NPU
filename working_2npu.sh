#!/bin/bash

echo "🎯 无初始评估的2NPU训练..."

# 基础环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 内存设置
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256"

# 清理
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 3

echo "基础分布式通信已验证成功✅"
echo "现在测试无初始评估的2NPU训练..."
echo ""
echo "配置:"
echo "  - 跳过初始验证评估"
echo "  - 使用与单NPU成功配置类似的参数"
echo "  - depth=4, batch=2"

# 参考成功的单NPU配置，但使用2NPU分布式
torchrun --standalone --nproc_per_node=2 -- scripts/base_train.py \
    --depth=4 \
    --device_batch_size=2 \
    --total_batch_size=16384 \
    --max_seq_len=512 \
    --num_iterations=10 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="no_eval_2npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 2NPU训练成功！现在可以尝试4NPU"
    echo "推荐下一步运行: ./working_4npu.sh"
else
    echo ""
    echo "❌ 2NPU训练失败"
    echo "建议运行: ./debug_2npu.sh 获取详细日志"
fi

echo ""
echo "2NPU测试完成: $(date)"