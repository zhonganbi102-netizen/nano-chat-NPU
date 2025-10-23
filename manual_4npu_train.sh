#!/bin/bash

# 手动4NPU分布式训练 - 最简单直接的方案
# 通过修改参数名避开torchrun冲突

set -e

echo "=== 手动4NPU分布式训练 ==="

# 1. 清理环境
if [ -f "./emergency_npu_cleanup.sh" ]; then
  bash ./emergency_npu_cleanup.sh
  sleep 5
fi

# 2. 设置环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29506
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 3. 创建补丁
cat > temp_manual_patch.py << EOF
import torch
from nanochat.gpt import GPT

def manual_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("手动NPU优化器设置: 全部AdamW")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    if embedding_params:
        opts.append(torch.optim.AdamW([{'params': embedding_params, 'lr': embedding_lr*0.8, 'initial_lr': embedding_lr*0.8}], lr=embedding_lr*0.8, weight_decay=weight_decay, betas=(0.9, 0.95)))
    if other_params:
        opts.append(torch.optim.AdamW([{'params': other_params, 'lr': matrix_lr*0.8, 'initial_lr': matrix_lr*0.8}], lr=matrix_lr*0.8, weight_decay=0.0, betas=(0.9, 0.95)))
    return opts

GPT.setup_optimizers = manual_optimizers
print("✅ 手动优化器补丁已应用")
EOF

# 4. 启动训练 - 避开--run参数冲突，使用--experiment_name
echo "4. 启动手动4NPU训练..."
python -c "import temp_manual_patch" && \
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29506 \
    scripts/base_train.py \
    --model_tag=manual_4npu \
    --depth=6 \
    --device_batch_size=2 \
    --total_batch_size=32768 \
    --num_iterations=500 \
    --embedding_lr=0.005 \
    --unembedding_lr=0.0005 \
    --matrix_lr=0.0025 \
    --grad_clip=0.5 \
    --eval_every=50 \
    --sample_every=250

# 5. 清理
rm -f temp_manual_patch.py

echo "手动4NPU训练完成！"
