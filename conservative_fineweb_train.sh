#!/bin/bash

# 保守内存配置的FineWeb训练脚本
# 专门针对NPU内存限制优化

set -e

echo "=== 保守内存FineWeb训练 ==="

# 1. 强力清理NPU环境
echo "1. 强力清理NPU内存..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
    ./emergency_npu_cleanup.sh
    sleep 10
fi

# 2. 设置环境
echo "2. 设置NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. 保守的4NPU环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29521
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64  # NPU安全的内存分割配置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export NPU_CALCULATE_DEVICE=0,1,2,3

# 4. 创建内存友好的优化器补丁
echo "3. 创建内存友好优化器补丁..."
cat > temp_conservative_patch.py << EOF
import torch
from nanochat.gpt import GPT

def conservative_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("保守内存优化器: AdamW + 低内存配置")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    
    # 使用更保守的学习率
    if embedding_params:
        opts.append(torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr*0.5, 'initial_lr': embedding_lr*0.5}], 
            lr=embedding_lr*0.5, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.95),
            eps=1e-6,
            foreach=False  # 关闭foreach优化以节省内存
        ))
    
    if other_params:
        opts.append(torch.optim.AdamW(
            [{'params': other_params, 'lr': matrix_lr*0.5, 'initial_lr': matrix_lr*0.5}], 
            lr=matrix_lr*0.5, 
            weight_decay=0.0, 
            betas=(0.9, 0.95),
            eps=1e-6,
            foreach=False  # 关闭foreach优化以节省内存
        ))
    
    return opts

GPT.setup_optimizers = conservative_optimizers
print("✅ 保守内存优化器补丁已应用")
EOF

# 5. 训练tokenizer
echo "4. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    python -m scripts.tok_train
else
    echo "tokenizer已存在"
fi

# 6. 保守内存训练配置
echo "5. 开始保守内存FineWeb训练..."
echo ""
echo "保守配置:"
echo "  - 模型深度: 8层 (降低内存)"
echo "  - 批次大小: 每设备4, 总共65536"
echo "  - 训练步数: 2000"
echo "  - 内存优化: 启用"
echo "  - 预计时间: 1-2小时"
echo ""

python -c "import temp_conservative_patch" && \
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29521 \
    scripts/base_train.py \
    --model_tag=fineweb_conservative_d8 \
    --depth=8 \
    --device_batch_size=4 \
    --total_batch_size=65536 \
    --num_iterations=2000 \
    --embedding_lr=0.1 \
    --unembedding_lr=0.002 \
    --matrix_lr=0.01 \
    --grad_clip=0.5 \
    --eval_every=200 \
    --sample_every=800 \
    --core_metric_every=999999

# 7. 清理
rm -f temp_conservative_patch.py

echo ""
echo "🎉 保守内存FineWeb训练完成！"
echo "模型保存: ~/.cache/nanochat/base_checkpoints/fineweb_conservative_d8/"
