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

# 4. 创建NPU兼容的优化器补丁（完全避免Muon）
echo "3. 创建NPU兼容优化器补丁..."
cat > npu_adamw_patch.py << EOF
import torch
from nanochat.gpt import GPT

def npu_compatible_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """
    NPU兼容的优化器 - 完全替代Muon
    使用标准AdamW，避免所有NPU不兼容的优化器
    """
    print("🔧 使用NPU兼容的AdamW优化器（替代Muon）")
    
    # 获取所有参数并分组
    embedding_params = []
    unembedding_params = []
    matrix_params = []
    
    for name, param in self.named_parameters():
        if 'emb_tok' in name:
            embedding_params.append(param)
            print(f"  Embedding参数: {name}, shape: {param.shape}")
        elif 'unembed' in name:
            unembedding_params.append(param)
            print(f"  Unembedding参数: {name}, shape: {param.shape}")
        else:
            matrix_params.append(param)
            print(f"  Matrix参数: {name}, shape: {param.shape}")
    
    optimizers = []
    
    # Embedding优化器
    if embedding_params:
        emb_opt = torch.optim.AdamW(
            embedding_params,
            lr=embedding_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,  # NPU兼容性
            fused=False     # 禁用fused优化
        )
        optimizers.append(emb_opt)
        print(f"  ✅ Embedding AdamW: lr={embedding_lr}, params={len(embedding_params)}")
    
    # Unembedding优化器
    if unembedding_params:
        unemb_opt = torch.optim.AdamW(
            unembedding_params,
            lr=unembedding_lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,  # NPU兼容性
            fused=False     # 禁用fused优化
        )
        optimizers.append(unemb_opt)
        print(f"  ✅ Unembedding AdamW: lr={unembedding_lr}, params={len(unembedding_params)}")
    
    # Matrix优化器
    if matrix_params:
        matrix_opt = torch.optim.AdamW(
            matrix_params,
            lr=matrix_lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,  # NPU兼容性
            fused=False     # 禁用fused优化
        )
        optimizers.append(matrix_opt)
        print(f"  ✅ Matrix AdamW: lr={matrix_lr}, params={len(matrix_params)}")
    
    print(f"🎯 总共创建了 {len(optimizers)} 个NPU兼容的AdamW优化器")
    return optimizers

# 替换原始的setup_optimizers方法
GPT.setup_optimizers = npu_compatible_optimizers
print("✅ NPU兼容优化器补丁已应用（完全避免Muon）")
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

python -c "import npu_adamw_patch" && \
PYTHONPATH=. torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29521 \
    scripts/base_train.py \
    --model_tag=fineweb_no_muon_d8 \
    --depth=8 \
    --device_batch_size=4 \
    --total_batch_size=65536 \
    --num_iterations=2000 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=0.8 \
    --eval_every=200 \
    --sample_every=800 \
    --core_metric_every=999999

# 7. 清理
rm -f npu_adamw_patch.py

echo ""
echo "🎉 保守内存FineWeb训练完成！"
echo "模型保存: ~/.cache/nanochat/base_checkpoints/fineweb_conservative_d8/"
