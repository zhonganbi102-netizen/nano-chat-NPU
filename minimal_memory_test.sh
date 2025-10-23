#!/bin/bash

# 极简内存测试脚本 - 最小配置验证NPU能力

set -e

echo "=== 极简内存测试 ==="

# 1. 强力清理
echo "1. 强力清理..."
./emergency_npu_cleanup.sh
sleep 15

# 2. 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29522
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:16  # 极小内存分割
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 3. 极简优化器
cat > temp_minimal_patch.py << EOF
import torch
from nanochat.gpt import GPT

def minimal_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("极简优化器: 纯AdamW")
    params = list(self.parameters())
    optimizer = torch.optim.AdamW(
        params, 
        lr=0.001,  # 固定学习率
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,
        fused=False
    )
    return [optimizer]

GPT.setup_optimizers = minimal_optimizers
print("✅ 极简优化器已应用")
EOF

# 4. 训练tokenizer
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    python -m scripts.tok_train
fi

# 5. 极简配置测试
echo "4. 极简配置测试..."
echo "配置: depth=6, batch_size=2, 50步"

python -c "import temp_minimal_patch" && \
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29522 \
    scripts/base_train.py \
    --model_tag=minimal_memory_test \
    --depth=6 \
    --device_batch_size=2 \
    --total_batch_size=32768 \
    --num_iterations=50 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=0.3 \
    --eval_every=25 \
    --sample_every=99999 \
    --core_metric_every=999999

rm -f temp_minimal_patch.py
echo "✅ 极简测试完成！"
