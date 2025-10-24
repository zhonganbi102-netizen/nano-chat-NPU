#!/bin/bash

# NPU兼容训练脚本 - 无Muon优化器
# NPU Compatible Training Script - Without Muon Optimizer

echo "🚀 启动NPU兼容训练（无Muon优化器）..."
echo "=================================================="

# 设置NPU环境变量
export NPU_VISIBLE_DEVICES=0
export HCCL_BUFFSIZE_MB=256
export TASK_QUEUE_ENABLE=1

# 检查NPU设备
echo "检查NPU设备状态..."
python3 -c "
import torch
try:
    import torch_npu
    if torch_npu.npu.is_available():
        print(f'✅ NPU可用: {torch_npu.npu.device_count()} 设备')
        for i in range(torch_npu.npu.device_count()):
            print(f'  设备 {i}: {torch_npu.npu.get_device_name(i)}')
    else:
        print('❌ NPU不可用')
except ImportError:
    print('❌ torch_npu未安装')
"

# 检查数据文件
if [ ! -d "fineweb" ]; then
    echo "⚠️  警告: fineweb目录不存在，创建示例数据..."
    mkdir -p fineweb
    python3 -c "
import json
import pandas as pd

# 创建示例数据
sample_data = [
    {'text': 'This is a sample text for training the language model.'},
    {'text': 'Natural language processing is a fascinating field of study.'},
    {'text': 'Machine learning models require large amounts of training data.'},
    {'text': 'The transformer architecture has revolutionized NLP.'},
    {'text': 'GPT models are based on the transformer decoder architecture.'}
] * 1000  # 重复创建更多数据

df = pd.DataFrame(sample_data)
df.to_parquet('fineweb/sample.parquet', index=False)
print('✅ 创建示例数据文件: fineweb/sample.parquet')
"
fi

# 设置Python路径
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "开始训练..."
echo "配置:"
echo "  - 优化器: AdamW (无Muon)"
echo "  - 设备: NPU"
echo "  - 批次大小: 4"
echo "  - 迭代次数: 500"
echo "  - 模型深度: 6"

# 运行训练
python3 single_npu_no_muon_train.py \
    --config_overrides "
run='npu_no_muon_test'
depth=6
device_batch_size=4
total_batch_size=8192
num_iterations=500
embedding_lr=0.001
unembedding_lr=0.0001
matrix_lr=0.0005
grad_clip=1.0
eval_every=100
sample_every=500
" 2>&1 | tee npu_no_muon_training.log

echo "=================================================="
echo "训练完成！日志保存在: npu_no_muon_training.log"
echo "如需4NPU分布式训练，请在验证单NPU训练成功后运行多卡版本"