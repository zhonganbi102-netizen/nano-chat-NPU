#!/bin/bash

# 单NPU极简内存测试 - 避免分布式训练的内存开销

set -e

echo "=== 单NPU极简内存测试 ==="

# 1. 强力清理
echo "1. 强力清理..."
./emergency_npu_cleanup.sh
sleep 15

# 2. 设置环境
echo "2. 设置单NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64  # 保守配置
export NPU_CALCULATE_DEVICE=0  # 只使用第一个NPU

# 3. 极简优化器
echo "3. 创建单NPU极简优化器..."
cat > temp_single_npu_patch.py << EOF
import torch
from nanochat.gpt import GPT

def single_npu_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("单NPU极简优化器: 纯AdamW + 最小内存")
    params = list(self.parameters())
    optimizer = torch.optim.AdamW(
        params, 
        lr=0.0005,  # 更小学习率
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
        fused=False,
        amsgrad=False
    )
    return [optimizer]

GPT.setup_optimizers = single_npu_optimizers
print("✅ 单NPU极简优化器已应用")
EOF

# 4. 训练tokenizer
echo "4. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    python -m scripts.tok_train
fi

# 5. 单NPU极简配置测试
echo "5. 单NPU极简配置测试..."
echo "配置: 单NPU, depth=4, batch_size=1, 20步"

python -c "import temp_single_npu_patch" && \
python -m scripts.base_train \
    --run=single_npu_memory_test \
    --depth=4 \
    --device_batch_size=1 \
    --total_batch_size=4096 \
    --num_iterations=20 \
    --embedding_lr=0.005 \
    --unembedding_lr=0.0005 \
    --matrix_lr=0.0025 \
    --grad_clip=0.2 \
    --eval_every=10 \
    --sample_every=99999 \
    --core_metric_every=999999

rm -f temp_single_npu_patch.py
echo "✅ 单NPU极简测试完成！"
