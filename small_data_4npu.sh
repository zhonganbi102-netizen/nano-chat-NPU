#!/bin/bash

echo "🧪 小数据多GPU测试..."

# 环境变量设置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
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
sleep 3

echo "配置信息："
echo "  🎯 4个NPU (0,1,2,3)"
echo "  📊 超小模型：depth=3, 每层很少参数"
echo "  💾 最小batch：device_batch_size=1"
echo "  📏 短序列：max_seq_len=256"
echo "  🔢 少步数：仅5步训练"
echo "  ⏭️  跳过所有评估，专注训练循环"
echo ""

echo "启动小数据4NPU测试..."

torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=3 \
    --device_batch_size=1 \
    --total_batch_size=4096 \
    --max_seq_len=256 \
    --num_iterations=5 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="small_data_4npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 小数据4NPU测试成功！"
    echo "现在可以逐步增加："
    echo "  1. 增加depth到6"
    echo "  2. 增加batch_size到2"  
    echo "  3. 增加训练步数到20"
else
    echo ""
    echo "❌ 小数据4NPU测试失败"
    echo "可能需要进一步减小配置"
fi

echo ""
echo "小数据测试完成: $(date)"