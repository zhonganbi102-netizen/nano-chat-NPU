#!/bin/bash

echo "=== 单NPU训练测试 ==="

# 只使用一张NPU，避免分布式问题
export ASCEND_RT_VISIBLE_DEVICES=0
WORLD_SIZE=1

echo "检查单NPU环境..."
python -c "
import torch_npu
print(f'可用NPU数量: {torch_npu.npu.device_count()}')
print(f'当前设备: {torch_npu.npu.current_device()}')
torch_npu.npu.set_device(0)
print(f'设备名称: {torch_npu.npu.get_device_name(0)}')

# 测试内存分配
try:
    x = torch_npu.zeros(1000, 1000)
    print(f'✅ 内存分配测试成功: {x.shape}')
    del x
    torch_npu.npu.empty_cache()
except Exception as e:
    print(f'❌ 内存分配测试失败: {e}')
"

echo ""
echo "开始单NPU训练测试..."

# 使用单NPU和更小的参数
python -m scripts.base_train \
    --run=npu_single_test \
    --depth=4 \
    --device_batch_size=4 \
    --total_batch_size=8192 \
    --num_iterations=5 \
    --eval_every=2 \
    --sample_every=10 \
    --core_metric_every=10

echo "单NPU测试完成"