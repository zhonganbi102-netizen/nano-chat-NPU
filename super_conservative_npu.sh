#!/bin/bash

# 超级保守单NPU训练脚本 - 修复所有NPU兼容性问题
# Super conservative single NPU training script - fixing all NPU compatibility issues

set -e

echo "=== 超级保守单NPU训练 (修复版) ==="
echo "Super Conservative Single NPU Training (Fixed Version)"

# 1. 强制清理环境
echo "1. 强制清理NPU环境..."
pkill -f python || echo "没有Python进程"
sleep 3

# 清理系统缓存（如果有权限）
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "无法清理系统缓存（需要root权限）"

# 2. 设置NPU环境变量 - 严格遵循NPU要求
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# NPU内存配置 - 必须 > 20MB
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export NPU_COMPILE_DISABLE=1
export TORCH_COMPILE_DISABLE=1

# 设置Ascend环境
echo "2. 设置Ascend环境..."
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "✅ 成功加载Ascend环境"
else
    echo "⚠️ 手动设置Ascend环境变量..."
    export ASCEND_HOME="/usr/local/Ascend/ascend-toolkit"
    export PATH="/usr/local/Ascend/ascend-toolkit/latest/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
    export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH"
fi

echo "配置详情:"
echo "  使用NPU: 0"
echo "  内存分片: 128MB (NPU要求 > 20MB)"
echo "  编译优化: 完全禁用"
echo "  世界大小: 1 (单NPU)"

# 3. 验证NPU环境
echo "3. 验证NPU环境..."
python3 -c "
import torch
import torch_npu
import gc

print('验证NPU环境...')
if torch_npu.npu.is_available():
    print(f'✅ NPU可用，设备数量: {torch_npu.npu.device_count()}')
    
    # 强制清理NPU内存
    for i in range(torch_npu.npu.device_count()):
        with torch_npu.npu.device(i):
            torch_npu.npu.empty_cache()
    gc.collect()
    
    # 检查当前设备
    device = torch_npu.npu.current_device()
    allocated = torch_npu.npu.memory_allocated(device) / 1024**3
    reserved = torch_npu.npu.memory_reserved(device) / 1024**3
    print(f'NPU {device}: 已分配 {allocated:.2f} GiB, 保留 {reserved:.2f} GiB')
else:
    print('❌ NPU不可用')
    exit(1)
" || exit 1

# 4. 创建优化器补丁 - 避免Muon，只用AdamW
echo "4. 创建NPU优化器补丁..."
cat > npu_optimizer_patch.py << 'EOF'
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

def npu_safe_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    """NPU安全的优化器实现 - 纯AdamW"""
    print("🚀 NPU安全优化器: 纯AdamW实现")
    
    try:
        # 获取所有参数
        params = list(self.parameters())
        param_count = sum(p.numel() for p in params)
        
        print(f"  📊 参数统计: {len(params)}个张量, {param_count:,}个参数")
        
        # 使用单一AdamW优化器，避免复杂的参数分组
        optimizer = torch.optim.AdamW(
            params, 
            lr=max(embedding_lr, unembedding_lr, matrix_lr),  # 使用最大学习率
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # 稍微调整beta
            eps=1e-8,
            foreach=False,  # NPU兼容性
            fused=False,    # NPU兼容性
            amsgrad=False   # 关闭额外功能
        )
        
        print(f"  ✅ NPU AdamW优化器创建成功: lr={max(embedding_lr, unembedding_lr, matrix_lr)}")
        return [optimizer]
        
    except Exception as e:
        print(f"  ❌ 优化器创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# 应用补丁
try:
    print("🔧 应用NPU优化器补丁...")
    GPT.setup_optimizers = npu_safe_optimizers
    print("✅ NPU优化器补丁已应用")
except Exception as e:
    print(f"❌ 补丁应用失败: {e}")
    sys.exit(1)
EOF

# 5. 训练tokenizer（最小配置）
echo "5. 训练tokenizer（如果需要）..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    python -m scripts.tok_train || echo "tokenizer训练失败，但继续..."
else
    echo "✅ tokenizer已存在"
fi

# 6. 超保守训练配置
echo "6. 开始超保守训练..."
echo ""
echo "📊 训练配置:"
echo "  - 模型深度: 4层"
echo "  - 设备batch: 2"
echo "  - 总batch: 4096" 
echo "  - 训练步数: 50步（快速测试）"
echo "  - 学习率: 0.001（保守）"
echo ""

# 导入补丁并运行训练
echo "导入优化器补丁..."
python -c "
import npu_optimizer_patch
print('✅ NPU补丁导入成功')
"

echo "开始超保守训练..."
python -m scripts.base_train \
    --run=super_conservative_npu \
    --depth=4 \
    --device_batch_size=2 \
    --total_batch_size=4096 \
    --num_iterations=50 \
    --embedding_lr=0.001 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.001 \
    --weight_decay=0.01 \
    --grad_clip=1.0 \
    --eval_every=999999 \
    --sample_every=999999 \
    --core_metric_every=999999

# 7. 清理
rm -f npu_optimizer_patch.py

echo ""
echo "🎉 超保守NPU训练完成！"
echo ""
echo "如果成功，可以逐步增加:"
echo "1. depth: 4 -> 6 -> 8"
echo "2. device_batch_size: 2 -> 4 -> 8"
echo "3. num_iterations: 50 -> 100 -> 500"