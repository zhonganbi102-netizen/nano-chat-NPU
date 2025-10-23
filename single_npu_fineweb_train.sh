#!/bin/bash

# 单NPU FineWeb训练 - 完全避免torchrun环境继承问题
# 基于成功配置的单NPU版本

set -e

echo "🚀 单NPU FineWeb训练 - 避免torchrun问题 🚀"

# 1. 强力清理
echo "1. 强力清理NPU环境..."
./emergency_npu_cleanup.sh
sleep 15

# 2. 设置环境 (单NPU)
echo "2. 设置单NPU环境..."

# 动态查找并设置环境
./find_ascend_env.sh
if [ -f ".ascend_env_path" ]; then
    source .ascend_env_path
    echo "✅ 使用set_env.sh: $ASCEND_SET_ENV_PATH"
    source "$ASCEND_SET_ENV_PATH"
    export ASCEND_HOME="$(dirname "$ASCEND_SET_ENV_PATH")"
else
    echo "⚠️ 手动设置环境变量..."
    export ASCEND_HOME="/usr/local/Ascend/ascend-toolkit"
    export PATH="/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
    export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH"
fi

# 确保关键路径
export PYTHONPATH="$ASCEND_HOME/python/site-packages:$PYTHONPATH"
export PYTHONPATH="$ASCEND_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export PYTHONPATH=".:$PYTHONPATH"

# 单NPU环境变量
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128  # 更大内存分配
export NPU_CALCULATE_DEVICE=0  # 只使用第一个NPU

echo "✅ 单NPU环境设置完成"

# 3. 验证环境
echo "3. 验证NPU环境..."
if python3 -c "import torch_npu; print('✅ torch_npu可用')" 2>/dev/null; then
    echo "✅ torch_npu验证成功"
else
    echo "❌ torch_npu验证失败"
    exit 1
fi

if python3 -c "import tbe; print('✅ TBE模块可用')" 2>/dev/null; then
    echo "✅ TBE验证成功"
else
    echo "⚠️ TBE验证失败，但继续尝试..."
fi

# 4. 单NPU优化器补丁
echo "4. 创建单NPU优化器补丁..."
cat > temp_single_npu_patch.py << 'EOF'
import torch
from nanochat.gpt import GPT

def single_npu_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("🚀 单NPU FineWeb优化器: 避免分布式复杂性")
    
    # 获取所有参数
    params = list(self.parameters())
    
    # 单一AdamW优化器，简化配置
    optimizer = torch.optim.AdamW(
        params, 
        lr=0.001,  # 固定学习率
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        foreach=False,  # 关闭foreach
        fused=False     # 关闭fused
    )
    
    print(f"  ✅ 单NPU优化器: lr=0.001, {len(params)}个参数")
    return [optimizer]

# 应用补丁
GPT.setup_optimizers = single_npu_optimizers
print("✅ 单NPU FineWeb优化器补丁已应用")
EOF

# 5. 训练tokenizer
echo "5. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    python -m scripts.tok_train
else
    echo "✅ tokenizer已存在"
fi

# 6. 单NPU FineWeb训练
echo ""
echo "🚀 启动单NPU FineWeb训练..."
echo ""
echo "📊 训练配置:"
echo "  - 单NPU训练 (避免分布式)"
echo "  - 模型深度: 8层"
echo "  - 批次大小: 8 (单NPU)"
echo "  - 总批次: 16384"
echo "  - 训练步数: 500步"
echo "  - 预计时间: 15-30分钟"
echo ""

# 直接运行base_train.py (无torchrun)
python -c "import temp_single_npu_patch" && \
python -m scripts.base_train \
    --run=single_npu_fineweb_d8 \
    --depth=8 \
    --device_batch_size=8 \
    --total_batch_size=16384 \
    --num_iterations=500 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=0.5 \
    --eval_every=100 \
    --sample_every=250 \
    --core_metric_every=999999

# 7. 清理
rm -f temp_single_npu_patch.py

echo ""
echo "🎉 单NPU FineWeb训练完成！"
echo ""
echo "📍 模型位置: ~/.cache/nanochat/base_checkpoints/single_npu_fineweb_d8/"
echo ""
echo "🔧 如果成功，可以考虑扩展到多NPU训练"
