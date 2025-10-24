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

# 4. 单NPU优化器补丁 - 更安全的版本
echo "4. 创建单NPU优化器补丁..."
cat > temp_single_npu_patch.py << 'EOF'
import torch
import sys
import os

# 确保可以找到nanochat模块
sys.path.insert(0, '.')

print("🔧 导入nanochat模块...")
try:
    from nanochat.gpt import GPT
    print("✅ GPT类导入成功")
except Exception as e:
    print(f"❌ GPT类导入失败: {e}")
    sys.exit(1)

def single_npu_optimizers_safe(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """安全的单NPU优化器实现"""
    print("🚀 单NPU FineWeb优化器: 纯AdamW实现")
    
    try:
        # 获取所有参数
        params = list(self.parameters())
        param_count = sum(p.numel() for p in params)
        
        print(f"  📊 参数统计: {len(params)}个张量, {param_count:,}个参数")
        
        # 使用最基础的AdamW配置
        optimizer = torch.optim.AdamW(
            params, 
            lr=0.0005,      # 保守的学习率
            weight_decay=0.01,
            betas=(0.9, 0.999),  # 标准beta值
            eps=1e-8,
            foreach=False,  # NPU兼容性
            fused=False,    # NPU兼容性
            amsgrad=False   # 关闭额外功能
        )
        
        print(f"  ✅ AdamW优化器创建成功: lr=0.0005")
        return [optimizer]
        
    except Exception as e:
        print(f"  ❌ 优化器创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# 安全应用补丁
try:
    print("🔧 应用单NPU优化器补丁...")
    GPT.setup_optimizers = single_npu_optimizers_safe
    print("✅ 单NPU FineWeb优化器补丁已应用")
except Exception as e:
    print(f"❌ 补丁应用失败: {e}")
    sys.exit(1)
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
echo "  - 模型深度: 6层 (更保守)"
echo "  - 批次大小: 4 (更小batch)"
echo "  - 总批次: 8192 (更小)"
echo "  - 训练步数: 100步 (测试)"
echo "  - 预计时间: 5-10分钟"
echo ""

# 导入补丁并运行训练
echo "导入优化器补丁..."
python -c "
import temp_single_npu_patch
print('✅ 补丁导入成功')
"

echo "开始训练..."
python -m scripts.base_train single_npu_config.py

# 7. 清理
rm -f temp_single_npu_patch.py

echo ""
echo "🎉 单NPU FineWeb训练完成！"
echo ""
echo "📍 模型位置: ~/.cache/nanochat/base_checkpoints/single_npu_fineweb_d8/"
echo ""
echo "🔧 如果成功，可以考虑扩展到多NPU训练"
