#!/bin/bash

# 最终修复版4NPU分布式训练脚本
# 解决configurator.py参数解析问题

set -e

echo "=== 最终修复版4NPU分布式训练脚本 ==="

# 1. 清理环境
echo "1. 清理环境..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
  bash ./emergency_npu_cleanup.sh
  sleep 10
else
  echo "未找到清理脚本，跳过清理步骤"
fi

# 2. 设置分布式环境变量
echo "2. 设置分布式环境变量..."
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,7  # 使用4个NPU
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503  # 使用新端口
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32

# HCCL设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export HCCL_BUFFSIZE=220
export HCCL_OP_BASE_FFTS_MODE_ENABLE=FALSE

# 3. 检查NPU状态
echo "3. 检查NPU状态..."
npu-smi info

# 4. 创建补丁脚本
echo "4. 创建补丁脚本..."
cat > ./temp_final_patch.py << EOL
import sys
import torch
from nanochat.gpt import GPT

def final_setup_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """最终版优化器设置 - 4NPU分布式稳定版"""
    print("使用最终版NPU兼容优化器设置 (4NPU): 全部AdamW")
    
    # 获取所有参数并简单分组
    embedding_params = []
    other_params = []
    
    for name, param in self.named_parameters():
        if 'emb_tok' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    # 创建优化器列表
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

# 应用补丁
GPT.setup_optimizers = final_setup_optimizers
print("✅ 最终版4NPU分布式优化器补丁已应用")
EOL

# 5. 开始4NPU训练 - 使用正确的参数格式
echo "5. 开始最终版4NPU分布式训练..."
python -c "import sys; sys.path.insert(0, '.'); import temp_final_patch" && \
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29503 \
    --nnodes=1 \
    --node_rank=0 \
    scripts/base_train.py \
    --run=npu_4gpu_final \
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
rm -f ./temp_final_patch.py

echo "最终版4NPU分布式训练完成！"
