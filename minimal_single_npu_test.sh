#!/bin/bash

# 极简单NPU测试 - 调试卡死问题
# Minimal single NPU test - debug hanging issues

set -e

echo "🔍 极简单NPU测试 - 调试模式"
echo ""

# 1. 设置环境
echo "1. 设置环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
export NPU_COMPILE_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTHONUNBUFFERED=1

echo "✅ 环境设置完成"

# 2. 清理NPU
echo "2. 清理NPU..."
python3 -c "
import torch
import torch_npu
if torch_npu.npu.is_available():
    torch_npu.npu.empty_cache()
    print('✅ NPU清理完成')
"

# 3. 创建最简优化器补丁
echo "3. 创建最简优化器补丁..."
cat > minimal_patch.py << 'EOF'
import torch
import sys
sys.path.insert(0, '.')

print("导入GPT...")
from nanochat.gpt import GPT

def minimal_optimizer(self, **kwargs):
    """最简优化器实现"""
    print("🔧 最简AdamW优化器")
    
    params = list(self.parameters())
    print(f"参数数量: {len(params)}")
    
    # 最基础的AdamW
    optimizer = torch.optim.AdamW(params, lr=0.001)
    print("✅ 优化器创建成功")
    
    return [optimizer]

# 替换方法
print("应用补丁...")
GPT.setup_optimizers = minimal_optimizer
print("✅ 补丁应用成功")
EOF

# 4. 运行最小测试
echo "4. 运行最小测试..."
python3 -c "
import minimal_patch
print('✅ 补丁加载成功')
"

echo "5. 开始极简训练..."
timeout 300 python -m scripts.base_train \
    --run=minimal_test \
    --depth=2 \
    --device_batch_size=1 \
    --total_batch_size=2 \
    --num_iterations=5 \
    --embedding_lr=0.001 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.001 \
    --eval_every=999999 \
    --sample_every=999999 \
    --core_metric_every=999999 \
    --verbose

echo ""
if [ $? -eq 0 ]; then
    echo "🎉 极简测试成功！"
else
    echo "❌ 测试失败或超时"
fi

# 清理
rm -f minimal_patch.py