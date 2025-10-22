#!/bin/bash

echo "=== 简化NPU训练测试 ==="

# NPU环境设置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=4

echo "检查NPU环境..."
python -c "
import torch_npu
print(f'NPU数量: {torch_npu.npu.device_count()}')
for i in range(min(4, torch_npu.npu.device_count())):
    print(f'NPU {i}: {torch_npu.npu.get_device_name(i)}')
"

echo ""
echo "开始简化测试训练..."

# 使用更小的参数进行测试
torchrun --standalone --nproc_per_node=$WORLD_SIZE -m scripts.base_train -- \
    --run=npu_test_small \
    --depth=6 \
    --device_batch_size=8 \
    --total_batch_size=131072 \
    --num_iterations=10 \
    --eval_every=5 \
    --sample_every=5 \
    --core_metric_every=10

echo "测试训练完成"