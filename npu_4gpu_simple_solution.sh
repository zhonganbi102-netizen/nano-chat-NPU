#!/bin/bash

# 简单直接的4NPU分布式训练脚本
# 通过环境变量传递参数，完全避开torchrun冲突

set -e

echo "=== 简单直接4NPU分布式训练脚本 ==="

# 1. 清理环境
echo "1. 清理环境..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
  bash ./emergency_npu_cleanup.sh
  sleep 10
fi

# 2. 设置分布式环境变量
echo "2. 设置分布式环境变量..."
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,7
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29505
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 3. 创建补丁
echo "3. 创建补丁..."
cat > temp_simple_patch.py << EOF
import torch
from nanochat.gpt import GPT

def simple_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("使用简单NPU兼容优化器 (4NPU): 全部AdamW")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    optimizers = []
    if embedding_params:
        optimizers.append(torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr * 0.8, 'initial_lr': embedding_lr * 0.8}],
            lr=embedding_lr * 0.8, weight_decay=weight_decay, betas=(0.9, 0.95)
        ))
    if other_params:
        optimizers.append(torch.optim.AdamW(
            [{'params': other_params, 'lr': matrix_lr * 0.8, 'initial_lr': matrix_lr * 0.8}],
            lr=matrix_lr * 0.8, weight_decay=0.0, betas=(0.9, 0.95) 
        ))
    return optimizers

GPT.setup_optimizers = simple_optimizers
print("✅ 简单NPU优化器补丁已应用")
EOF

# 4. 创建简化版训练脚本，直接设置变量而不是命令行参数
echo "4. 创建简化训练脚本..."
cat > train_simple_4npu.py << EOF
import os
import sys
sys.path.insert(0, '.')
import temp_simple_patch

# 直接修改全局变量，避开命令行参数解析
import scripts.base_train

# 在导入后修改全局变量
scripts.base_train.run = "npu_4gpu_simple"
scripts.base_train.depth = 6
scripts.base_train.device_batch_size = 2
scripts.base_train.total_batch_size = 32768
scripts.base_train.num_iterations = 500
scripts.base_train.embedding_lr = 0.005
scripts.base_train.unembedding_lr = 0.0005
scripts.base_train.matrix_lr = 0.0025
scripts.base_train.grad_clip = 0.5
scripts.base_train.eval_every = 50
scripts.base_train.sample_every = 250

print("✅ 参数设置完成，开始训练...")
EOF

# 5. 开始训练
echo "5. 开始简单4NPU分布式训练..."
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29505 \
    --nnodes=1 \
    --node_rank=0 \
    train_simple_4npu.py

# 6. 清理
rm -f temp_simple_patch.py train_simple_4npu.py

echo "简单4NPU分布式训练完成！"
