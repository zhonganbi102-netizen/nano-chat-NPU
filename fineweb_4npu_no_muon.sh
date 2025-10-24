#!/bin/bash

# 4NPU FineWeb训练脚本 - 完全无Muon版本
# 4NPU FineWeb training script - completely Muon-free version

set -e

echo "🚀 4NPU FineWeb训练 - 无Muon版本"
echo ""

# 1. 环境设置
echo "1. 设置4NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 4NPU分布式环境
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29900
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1
export NPU_CALCULATE_DEVICE=0,1,2,3
export OMP_NUM_THREADS=1

echo "✅ 4NPU环境配置完成"

# 2. 强力清理
echo ""
echo "2. 强力清理NPU环境..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
    ./emergency_npu_cleanup.sh
    sleep 5
fi

# 3. 创建4NPU无Muon优化器补丁
echo ""
echo "3. 创建4NPU无Muon优化器补丁..."
cat > fineweb_4npu_no_muon_patch.py << 'EOF'
import torch
from nanochat.gpt import GPT

def fineweb_4npu_adamw_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """
    4NPU FineWeb专用AdamW优化器 - 完全替代Muon
    针对大规模训练优化
    """
    print("🔧 4NPU FineWeb: 使用纯AdamW优化器（无Muon，分布式友好）")
    
    # 详细参数分组
    embedding_params = []
    unembedding_params = []
    attention_params = []
    ffn_params = []
    layernorm_params = []
    other_params = []
    
    for name, param in self.named_parameters():
        if 'emb_tok' in name:
            embedding_params.append(param)
        elif 'unembed' in name:
            unembedding_params.append(param)
        elif 'attn' in name:
            attention_params.append(param)
        elif 'ffn' in name or 'mlp' in name:
            ffn_params.append(param)
        elif 'norm' in name or 'ln' in name:
            layernorm_params.append(param)
        else:
            other_params.append(param)
    
    print(f"参数分组统计:")
    print(f"  Embedding: {len(embedding_params)} 参数")
    print(f"  Unembedding: {len(unembedding_params)} 参数")
    print(f"  Attention: {len(attention_params)} 参数")
    print(f"  FFN: {len(ffn_params)} 参数")
    print(f"  LayerNorm: {len(layernorm_params)} 参数")
    print(f"  Other: {len(other_params)} 参数")
    
    optimizers = []
    
    # Embedding优化器（高学习率）
    if embedding_params:
        emb_opt = torch.optim.AdamW(
            embedding_params,
            lr=embedding_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        optimizers.append(emb_opt)
        print(f"  ✅ Embedding AdamW: lr={embedding_lr}")
    
    # Unembedding优化器（低学习率）
    if unembedding_params:
        unemb_opt = torch.optim.AdamW(
            unembedding_params,
            lr=unembedding_lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        optimizers.append(unemb_opt)
        print(f"  ✅ Unembedding AdamW: lr={unembedding_lr}")
    
    # Attention和FFN参数（中等学习率）
    matrix_params = attention_params + ffn_params + other_params
    if matrix_params:
        matrix_opt = torch.optim.AdamW(
            matrix_params,
            lr=matrix_lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        optimizers.append(matrix_opt)
        print(f"  ✅ Matrix AdamW: lr={matrix_lr}")
    
    # LayerNorm参数（低学习率，无weight decay）
    if layernorm_params:
        ln_opt = torch.optim.AdamW(
            layernorm_params,
            lr=matrix_lr * 0.5,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        optimizers.append(ln_opt)
        print(f"  ✅ LayerNorm AdamW: lr={matrix_lr * 0.5}")
    
    print(f"🎯 总共创建了 {len(optimizers)} 个分布式友好的AdamW优化器")
    return optimizers

# 替换原始方法
GPT.setup_optimizers = fineweb_4npu_adamw_optimizers
print("✅ 4NPU FineWeb无Muon优化器补丁已应用")
EOF

# 4. 检查tokenizer
echo ""
echo "4. 检查tokenizer..."
if [ ! -f "tokenizer/tokenizer.json" ] && [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    python -m scripts.tok_train
else
    echo "✅ tokenizer已存在"
fi

# 5. 开始4NPU FineWeb训练
echo ""
echo "5. 开始4NPU FineWeb训练（无Muon）..."
echo ""
echo "配置:"
echo "  - 模型深度: 10层"
echo "  - 批次大小: 每设备6, 总98304"
echo "  - 训练步数: 3000步"
echo "  - 优化器: 纯AdamW（分布式友好）"
echo "  - 预计时间: 2-3小时"
echo ""

python3 -c "import fineweb_4npu_no_muon_patch" && \
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29900 \
    scripts/base_train.py \
    --model_tag=fineweb_4npu_no_muon_d10 \
    --depth=10 \
    --device_batch_size=6 \
    --total_batch_size=98304 \
    --num_iterations=3000 \
    --embedding_lr=0.02 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.008 \
    --grad_clip=0.8 \
    --eval_every=300 \
    --sample_every=900 \
    --core_metric_every=999999 \
    --verbose

# 6. 清理
rm -f fineweb_4npu_no_muon_patch.py

echo ""
echo "🎉 4NPU FineWeb无Muon训练完成！"
echo "模型保存: ~/.cache/nanochat/base_checkpoints/fineweb_4npu_no_muon_d10/"