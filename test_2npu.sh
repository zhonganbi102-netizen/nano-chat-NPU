#!/bin/bash

echo "🧪 2NPU测试训练..."

# 使用2个NPU进行更简单的测试
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

# 清理NPU缓存
python3 -c "
import torch_npu
for i in range(2):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    except:
        pass
"

echo "启动2NPU测试..."
echo "配置: depth=4, batch=2, 仅10步训练"

torchrun --standalone --nproc_per_node=2 -- scripts/base_train.py \
    --depth=4 \
    --device_batch_size=2 \
    --total_batch_size=16384 \
    --max_seq_len=512 \
    --num_iterations=10 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="2npu_test_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo "✅ 2NPU测试成功！可以尝试4NPU训练"
else
    echo "❌ 2NPU测试失败，需要进一步调试"
fi

echo "测试完成: $(date)"