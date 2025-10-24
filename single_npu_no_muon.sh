#!/bin/bash

# 单NPU训练脚本 - 完全避免Muon优化器
# Single NPU training script - completely avoid Muon optimizer

set -e

echo "🚀 单NPU训练 - 无Muon版本"
echo ""

# 1. 环境设置
echo "1. 设置单NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单NPU环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29800

# NPU优化设置
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export NPU_COMPILE_DISABLE=1
export TORCH_NPU_DISABLE_LAZY_INIT=1
export OMP_NUM_THREADS=1

echo "✅ 环境配置完成"

# 2. 清理NPU
echo ""
echo "2. 清理NPU环境..."
pkill -f "python.*train" || true
pkill -f "torchrun" || true

python3 -c "
import torch
import torch_npu
import gc
if torch_npu.npu.is_available():
    torch_npu.npu.empty_cache()
    gc.collect()
    print('✅ NPU缓存已清理')
"

# 3. 创建无Muon优化器补丁
echo ""
echo "3. 创建无Muon优化器补丁..."
cat > single_npu_no_muon_patch.py << 'EOF'
import torch
from nanochat.gpt import GPT

def single_npu_adamw_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """
    单NPU专用AdamW优化器 - 完全替代Muon
    """
    print("🔧 单NPU: 使用纯AdamW优化器（无Muon）")
    
    # 获取参数
    all_params = list(self.parameters())
    print(f"总参数数量: {len(all_params)}")
    print(f"总参数量: {sum(p.numel() for p in all_params):,}")
    
    # 简化参数分组
    embedding_params = []
    other_params = []
    
    for name, param in self.named_parameters():
        if 'emb_tok' in name:
            embedding_params.append(param)
        else:
            other_params.append(param)
    
    print(f"Embedding参数: {len(embedding_params)}")
    print(f"其他参数: {len(other_params)}")
    
    optimizers = []
    
    # 只使用标准AdamW，完全避免复杂优化器
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
        print(f"✅ Embedding AdamW: lr={embedding_lr}")
    
    if other_params:
        other_opt = torch.optim.AdamW(
            other_params,
            lr=matrix_lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        optimizers.append(other_opt)
        print(f"✅ Other AdamW: lr={matrix_lr}")
    
    print(f"🎯 创建了 {len(optimizers)} 个AdamW优化器")
    return optimizers

# 替换优化器方法
GPT.setup_optimizers = single_npu_adamw_optimizers
print("✅ 单NPU无Muon优化器补丁已应用")
EOF

# 4. 开始训练
echo ""
echo "4. 开始单NPU训练（无Muon）..."
echo "配置:"
echo "  - 模型深度: 6层"
echo "  - 批次大小: 4"
echo "  - 训练步数: 500步"
echo "  - 优化器: 纯AdamW"
echo ""

python3 -c "import single_npu_no_muon_patch" && \
python3 -m scripts.base_train \
    --run=single_npu_no_muon \
    --depth=6 \
    --device_batch_size=4 \
    --total_batch_size=4 \
    --num_iterations=500 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=1.0 \
    --eval_every=100 \
    --sample_every=200 \
    --core_metric_every=999999 \
    --verbose

# 5. 清理
rm -f single_npu_no_muon_patch.py

echo ""
echo "🎉 单NPU无Muon训练完成！"
echo "如果成功，可以尝试增加参数："
echo "  bash conservative_fineweb_train.sh  # 4NPU版本"