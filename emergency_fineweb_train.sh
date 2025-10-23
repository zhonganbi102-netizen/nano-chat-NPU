#!/bin/bash

# 紧急FineWeb训练脚本 - 基于成功的manual_4npu_train.sh配置
# 专门解决TBE模块环境继承问题

set -e

echo "🚨 紧急FineWeb训练 - 基于成功配置 🚨"

# 1. 强力清理
echo "1. 强力清理NPU环境..."
./emergency_npu_cleanup.sh
sleep 20  # 更长等待时间

# 2. 完整环境设置 (基于成功配置)
echo "2. 设置完整NPU环境..."

# 动态查找set_env.sh
echo "🔍 查找set_env.sh文件..."
./find_ascend_env.sh
if [ -f ".ascend_env_path" ]; then
    source .ascend_env_path
    echo "✅ 找到set_env.sh: $ASCEND_SET_ENV_PATH"
    source "$ASCEND_SET_ENV_PATH"
    
    # 从set_env.sh路径推断ASCEND_HOME
    export ASCEND_HOME="$(dirname "$ASCEND_SET_ENV_PATH")"
    echo "✅ ASCEND_HOME: $ASCEND_HOME"
else
    echo "❌ 找不到set_env.sh，手动设置环境..."
    # 手动设置基本环境变量
    export ASCEND_HOME="/usr/local/Ascend/ascend-toolkit"
    export PATH="/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
    export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH"
fi

# 显式设置关键路径
export PYTHONPATH="$ASCEND_HOME/python/site-packages:$PYTHONPATH"
export PYTHONPATH="$ASCEND_HOME/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export PYTHONPATH=".:$PYTHONPATH"
export LD_LIBRARY_PATH="$ASCEND_HOME/lib64:$LD_LIBRARY_PATH"

# 分布式训练环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29524
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

echo "✅ 环境变量设置完成"

# 3. 验证TBE模块
echo "3. 验证TBE模块..."
if python3 -c "import tbe; print('✅ TBE模块可用')" 2>/dev/null; then
    echo "✅ TBE验证成功"
else
    echo "❌ TBE验证失败，但继续尝试..."
fi

# 4. 基于成功配置的优化器补丁
echo "4. 创建紧急优化器补丁..."
cat > temp_emergency_patch.py << 'EOF'
import torch
from nanochat.gpt import GPT

def emergency_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("🚨 紧急FineWeb训练优化器: 基于manual_4npu成功配置")
    
    # 获取所有参数
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    
    opts = []
    
    # 嵌入层优化器 (保守配置)
    if embedding_params:
        embedding_opt = torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr*0.4, 'initial_lr': embedding_lr*0.4}], 
            lr=embedding_lr*0.4, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        opts.append(embedding_opt)
        print(f"  ✅ 嵌入层优化器: lr={embedding_lr*0.4:.6f}")
    
    # 其他参数优化器
    if other_params:
        other_opt = torch.optim.AdamW(
            [{'params': other_params, 'lr': matrix_lr*0.4, 'initial_lr': matrix_lr*0.4}], 
            lr=matrix_lr*0.4, 
            weight_decay=0.0, 
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False
        )
        opts.append(other_opt)
        print(f"  ✅ 其他参数优化器: lr={matrix_lr*0.4:.6f}")
    
    print(f"  ✅ 总共 {len(opts)} 个优化器")
    return opts

# 应用补丁
GPT.setup_optimizers = emergency_optimizers
print("✅ 紧急FineWeb优化器补丁已应用")
EOF

# 5. 训练tokenizer
echo "5. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    python -m scripts.tok_train
else
    echo "✅ tokenizer已存在"
fi

# 6. 启动紧急FineWeb训练 (基于manual成功配置)
echo ""
echo "🚀 启动紧急FineWeb训练..."
echo ""
echo "📊 训练配置:"
echo "  - 基于: manual_4npu_train.sh 成功配置"
echo "  - 模型深度: 8层"
echo "  - 批次大小: 每设备2, 总32768 (保守内存)"
echo "  - 训练步数: 1000步"
echo "  - 环境: 完整TBE路径设置"
echo "  - 预计时间: 30-60分钟"
echo ""

# 显式导入补丁并启动训练
python -c "import temp_emergency_patch" && \
PYTHONPATH="$ASCEND_HOME/python/site-packages:$ASCEND_HOME/opp/built-in/op_impl/ai_core/tbe:." torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29524 \
    scripts/base_train.py \
    --model_tag=emergency_fineweb_d8 \
    --depth=8 \
    --device_batch_size=2 \
    --total_batch_size=32768 \
    --num_iterations=1000 \
    --embedding_lr=0.005 \
    --unembedding_lr=0.0005 \
    --matrix_lr=0.0025 \
    --grad_clip=0.5 \
    --eval_every=100 \
    --sample_every=500 \
    --core_metric_every=999999

# 7. 清理
rm -f temp_emergency_patch.py

echo ""
echo "🎉 紧急FineWeb训练完成！"
echo ""
echo "📍 模型位置: ~/.cache/nanochat/base_checkpoints/emergency_fineweb_d8/"
echo ""
echo "🔧 如果成功，可以基于此配置进行更大规模训练"
