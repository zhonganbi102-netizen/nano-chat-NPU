#!/bin/bash

# 保守的4NPU分布式训练脚本
# 使用更小的配置以确保稳定性

set -e

echo "=== 保守4NPU分布式训练脚本 ==="

# 1. 清理环境
echo "1. 清理环境..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
  bash ./emergency_npu_cleanup.sh
  sleep 10
else
  echo "未找到清理脚本，跳过清理步骤"
fi

# 2. 设置保守的分布式环境变量
echo "2. 设置保守的分布式环境变量..."
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,7  # 使用4个NPU
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501  # 使用不同端口避免冲突
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32  # 保守的内存设置

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_BUFFSIZE=220
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE

# 3. 检查NPU状态
echo "3. 检查NPU状态..."
npu-smi info

# 4. 创建保守的补丁脚本
echo "4. 创建保守的补丁脚本..."
cat > ./temp_conservative_patch.py << EOL
import sys
import torch
from nanochat.gpt import GPT

def conservative_setup_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """保守的优化器设置 - 确保4NPU分布式稳定性"""
    print("使用保守的NPU兼容优化器设置 (4NPU): 全部AdamW + 保守参数")
    
    # 获取所有参数
    all_params = list(self.parameters())
    
    # 简单分组 - 避免复杂的参数分组可能导致的问题
    embedding_params = []
    other_params = []
    
    for name, param in self.named_parameters():
        if 'emb_tok' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    # 创建两个优化器组
    optimizers = []
    
    if embedding_params:
        opt1 = torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr * 0.7, 'initial_lr': embedding_lr * 0.7}],
            lr=embedding_lr * 0.7, 
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        optimizers.append(opt1)
    
    if other_params:
        opt2 = torch.optim.AdamW(
            [{'params': other_params, 'lr': matrix_lr * 0.7, 'initial_lr': matrix_lr * 0.7}],
            lr=matrix_lr * 0.7,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        optimizers.append(opt2)
    
    return optimizers

# 应用保守补丁
GPT.setup_optimizers = conservative_setup_optimizers
print("✅ 保守4NPU分布式优化器补丁已应用")
EOL

# 5. 开始保守的4NPU训练
echo "5. 开始保守的4NPU分布式训练..."
python -c "import sys; sys.path.insert(0, '.'); import temp_conservative_patch" && \
torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29501 \
    --nnodes=1 \
    --node_rank=0 \
    -m scripts.base_train \
    --run=npu_4gpu_conservative \
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

# 6. 清理临时文件
rm -f ./temp_conservative_patch.py

echo "保守4NPU分布式训练完成！"
