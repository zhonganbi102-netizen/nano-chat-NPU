#!/bin/bash

# NPU简化训练脚本
# 使用AdamW替代Muon优化器

set -e

echo "=== NPU简化训练脚本 ==="

# 1. 清理环境
echo "1. 清理环境..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
  bash ./emergency_npu_cleanup.sh
  sleep 5
else
  echo "未找到清理脚本，跳过清理步骤"
fi

# 2. 设置环境变量
echo "2. 设置环境变量..."
export ASCEND_RT_VISIBLE_DEVICES=0  # 只使用一个NPU
export WORLD_SIZE=1
export TORCH_COMPILE_DISABLE=1  # 禁用编译

# 3. 创建补丁脚本
echo "3. 创建补丁脚本..."
cat > ./temp_npu_patch.py << EOL
import sys
import torch
from nanochat.gpt import GPT

# 保存原始的setup_optimizers方法
original_setup_optimizers = GPT.setup_optimizers

def patched_setup_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """全部使用AdamW替代Muon优化器"""
    print("使用NPU兼容的优化器设置: 全部AdamW")
    
    # 将所有参数分组
    params_dict = {n: p for n, p in self.named_parameters()}
    
    # 第一组：词嵌入参数
    embedding_params = []
    if hasattr(self, 'emb_tok') and hasattr(self.emb_tok, 'weight'):
        embedding_params.append(self.emb_tok.weight)
    
    # 第二组：其他所有参数
    matrix_params = []
    for n, p in self.named_parameters():
        if n != 'emb_tok.weight' and n != 'lm_head.weight':
            matrix_params.append(p)
    
    # 第三组：lm_head权重
    unembedding_params = []
    if hasattr(self, 'lm_head') and hasattr(self.lm_head, 'weight'):
        unembedding_params.append(self.lm_head.weight)
    
    # 创建AdamW优化器
    optimizers = []
    
    if embedding_params:
        adamw_embedding = torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr, 'initial_lr': embedding_lr}],
            lr=embedding_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        optimizers.append(adamw_embedding)
    
    if matrix_params:
        adamw_matrix = torch.optim.AdamW(
            [{'params': matrix_params, 'lr': matrix_lr, 'initial_lr': matrix_lr}],
            lr=matrix_lr, weight_decay=0.0, betas=(0.9, 0.95)
        )
        optimizers.append(adamw_matrix)
    
    if unembedding_params:
        adamw_unembedding = torch.optim.AdamW(
            [{'params': unembedding_params, 'lr': unembedding_lr, 'initial_lr': unembedding_lr}],
            lr=unembedding_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )
        optimizers.append(adamw_unembedding)
    
    return optimizers

# 应用补丁
GPT.setup_optimizers = patched_setup_optimizers
print("✅ NPU优化器补丁已应用")
EOL

# 4. 开始训练
echo "4. 开始训练..."
python -c "import sys; sys.path.insert(0, '.'); import temp_npu_patch" && \
python -m scripts.base_train \
    --run=npu_simple_train \
    --depth=6 \
    --device_batch_size=4 \
    --total_batch_size=16384 \
    --num_iterations=500 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=0.5

# 5. 清理临时文件
rm -f ./temp_npu_patch.py

echo "训练完成！"
