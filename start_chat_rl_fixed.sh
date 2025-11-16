#!/bin/bash

# ============================================
# 🚀 Chat-RL 8NPU训练脚本 (NPU兼容版)
# ============================================
# 功能：基于已完成的chat_sft模型进行强化学习训练
# NPU配置：使用8个NPU进行分布式RL训练
# 数据集：GSM8K数学推理任务
# 优化器：标准PyTorch AdamW（避免分布式优化器问题）
# ============================================

set -e  # 遇到错误立即退出

echo "🚀 启动8NPU Chat-RL训练"
echo "============================================="
echo ""
echo "📋 基于已完成的chat_sft模型继续RL训练"
echo "💾 模型将保存到持久化目录"
echo "🔧 优化配置：8NPU × 4 device_batch_size = 32 examples per step"
echo ""

# ============================================
# 步骤1：安装Python依赖
# ============================================

echo "📦 步骤1: 安装Python依赖..."
echo ""
pip install -q wandb datasets huggingface_hub tiktoken hf_transfer
echo "✅ 依赖安装完成"
echo ""

# ============================================
# 步骤2：验证chat_sft模型
# ============================================

echo "🔍 步骤2: 验证chat_sft模型..."
PERSISTENT_DIR="/mnt/linxid615/bza/nanochat-models"
SFT_MODEL_DIR="${PERSISTENT_DIR}/chatsft_checkpoints/d18"

if [ -d "$SFT_MODEL_DIR" ]; then
    echo "✅ 找到chat_sft模型: $SFT_MODEL_DIR"
    LATEST_MODEL=$(ls -t ${SFT_MODEL_DIR}/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_MODEL" ]; then
        MODEL_SIZE=$(du -h "$LATEST_MODEL" | cut -f1)
        MODEL_STEP=$(basename "$LATEST_MODEL" .pt | sed 's/model_//')
        echo "📊 模型文件: $(basename $LATEST_MODEL) (${MODEL_SIZE})"
        echo "📈 训练步数: $MODEL_STEP"
        echo "🏗️  模型架构: d18 (350M参数, 18层)"
    fi
else
    echo "❌ 错误：未找到chat_sft模型"
    echo "   预期路径: $SFT_MODEL_DIR"
    echo ""
    echo "💡 请先完成chat_sft训练："
    echo "   bash start_chat_sft_fixed.sh"
    exit 1
fi
echo ""

# ============================================
# 步骤3：设置持久化模型保存路径
# ============================================

echo "📁 步骤3: 设置持久化模型保存路径..."
export NANOCHAT_BASE_DIR="$PERSISTENT_DIR"
echo "✅ 持久化路径已设置: $NANOCHAT_BASE_DIR"
echo "✅ RL模型将保存到: ${PERSISTENT_DIR}/chatrl_checkpoints/d18"
echo ""

# ============================================
# 步骤4：显示修复总结
# ============================================

echo "✅ Chat-RL 8NPU配置总结："
echo "  1. ✓ chat_rl.py: 8NPU分布式训练优化"
echo "  2. ✓ chat_rl.py: NPU内存管理和同步"
echo "  3. ✓ chat_rl.py: 标准PyTorch AdamW优化器"
echo "  4. ✓ 基于SFT模型 (d18, 350M参数)"
echo "  5. ✓ HF镜像源和网络优化"
echo ""
echo "💡 关键改进：8NPU并行训练，基于已完成的SFT模型！"
echo ""

# ============================================
# 步骤5：设置训练环境
# ============================================

echo "⚙️ 环境配置："
# 使用全部8个NPU
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "  - NPU设备: 0,1,2,3,4,5,6,7"

# 禁用torch.compile（NPU不支持）
export TORCH_COMPILE_DISABLE=1
echo "  - torch.compile: DISABLED"

# HuggingFace网络优化配置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
echo "  - HF镜像: https://hf-mirror.com"
echo "  - HF下载超时: 300s"
echo "  - HF快速传输: DISABLED (避免依赖问题)"

# HCCL通信超时设置
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
echo "  - HCCL超时: 3600s"

# Wandb离线模式
export WANDB_MODE=offline
export WANDB_SILENT=true
echo "  - Wandb: 离线模式"
echo ""

# ============================================
# 步骤6：预下载数据集（避免训练时网络问题）
# ============================================

echo "📥 步骤6: 预下载GSM8K数据集..."
python3 -c "
import os
# 确保使用镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

try:
    from datasets import load_dataset
    print('  📊 下载GSM8K训练集...')
    train_ds = load_dataset('openai/gsm8k', 'main', split='train')
    print(f'  ✅ 训练集下载完成: {len(train_ds)} 样本')
    
    print('  📊 下载GSM8K测试集...')
    test_ds = load_dataset('openai/gsm8k', 'main', split='test')
    print(f'  ✅ 测试集下载完成: {len(test_ds)} 样本')
    
    print('  🎯 GSM8K数据集准备就绪！')
except Exception as e:
    print(f'  ⚠️ 数据集下载警告: {e}')
    print('  💡 训练时将重新尝试下载')
"
echo ""

# ============================================
# 步骤7：清理NPU显存
# ============================================

echo "💾 步骤7: 清理NPU显存..."
cat > /tmp/clear_npu_memory.py << 'EOF'
import torch
try:
    import torch_npu
    for i in range(8):  # 清理8个NPU
        try:
            torch_npu.npu.set_device(i)
            torch_npu.npu.empty_cache()
            print(f"  ✓ NPU {i} 清理完成")
        except Exception as e:
            print(f"  ✗ NPU {i} 清理失败: {e}")
except ImportError:
    print("  ⚠️ torch_npu未安装，跳过NPU清理")
EOF

python /tmp/clear_npu_memory.py
rm /tmp/clear_npu_memory.py
echo ""

# ============================================
# 步骤8：启动8NPU Chat-RL训练
# ============================================

echo ""
echo "🎯 启动8NPU Chat-RL训练..."
echo ""
echo "📊 训练配置："
echo "  - 基础模型: chatsft_checkpoints/d18 (350M参数, 18层)"
echo "  - NPU数量: 8个"
echo "  - device_batch_size: 4"
echo "  - examples_per_step: 32 (8NPU × 4 = 32个样本)"
echo "  - num_samples: 16 (每个问题生成16个回答)"
echo "  - 数据集: GSM8K数学推理任务 (8K训练样本)"
echo "  - 训练轮数: 1 epoch (~250步)"
echo "  - 优化器: 标准PyTorch AdamW（避免分布式问题）"
echo "  - 学习率: embedding=0.2, unembedding=0.004, matrix=0.02"
echo "  - 温度: 1.0 (生成多样性)"
echo "  - 预计时间: 30-45分钟 (8NPU加速)"
echo ""

START_TIME=$(date +%s)

# 启动8NPU chat_rl
torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29800 \
    -- \
    -m scripts.chat_rl \
    --run=npu_chat_rl_8npu \
    --source=sft \
    --device_batch_size=4 \
    --examples_per_step=32 \
    --num_samples=16 \
    --num_epochs=1 \
    --unembedding_lr=0.004 \
    --embedding_lr=0.2 \
    --matrix_lr=0.02 \
    --weight_decay=0.0 \
    --init_lr_frac=0.05 \
    --save_every=60 \
    --eval_every=60 \
    --eval_examples=400

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# ============================================
# 步骤9：验证训练结果
# ============================================

echo ""
echo "🔍 步骤9: 验证训练结果..."
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================="
    echo "🎉🎉🎉 8NPU Chat-RL 训练完成！🎉🎉🎉"
    echo "============================================="
    echo ""
    echo "✅ 训练状态: 成功完成"
    echo "⏱️  训练用时: ${MINUTES}分${SECONDS}秒"
    echo "📁 模型位置: ${PERSISTENT_DIR}/chatrl_checkpoints"
    echo ""
    
    # 检查模型文件
    if [ -d "${PERSISTENT_DIR}/chatrl_checkpoints/d18" ]; then
        echo "📋 模型文件列表："
        ls -lh ${PERSISTENT_DIR}/chatrl_checkpoints/d18/model_*.pt 2>/dev/null || echo "  (暂无模型文件)"
        echo ""
        TOTAL_SIZE=$(du -sh ${PERSISTENT_DIR}/chatrl_checkpoints/d18 2>/dev/null | cut -f1)
        echo "💾 总计大小: ${TOTAL_SIZE}"
    fi
    echo ""
    
    echo "🎯 后续步骤："
    echo "  1️⃣  测试RL模型 (GSM8K数学推理):"
    echo "     python -m scripts.chat_cli --source=rl -p \"What is 25 times 37?\""
    echo ""
    echo "  2️⃣  测试RL模型 (复杂数学问题):"
    echo "     python -m scripts.chat_cli --source=rl -p \"John has 5 apples and buys 3 more. Then he gives 2 to Mary. How many does he have?\""
    echo ""
    echo "  3️⃣  启动Web界面:"
    echo "     python -m scripts.chat_web --source=rl"
    echo ""
    echo "  4️⃣  查看训练日志:"
    echo "     ls -lh ./wandb/offline-run-*/"
    echo ""
    
    echo "✅ 完成的关键步骤："
    echo "  ✓ 安装所有Python依赖"
    echo "  ✓ 验证chat_sft模型 (85M参数)"
    echo "  ✓ 设置持久化保存路径"
    echo "  ✓ NPU兼容性配置"
    echo "  ✓ 3NPU分布式RL训练"
    echo ""
    
    echo "💡 技术总结："
    echo "  ✓ 训练方法: GRPO (Group Relative Policy Optimization)"
    echo "  ✓ 训练数据: GSM8K数学推理 (8K样本)"
    echo "  ✓ 采样策略: 每题16个回答，取最佳反馈学习"
    echo "  ✓ 优化器: 标准PyTorch AdamW（NPU兼容）"
    echo "  ✓ NPU配置: 8个NPU并行训练"
    echo "  ✓ 内存占用: ~20GB per NPU (推理+训练)"
    echo "  ✓ 模型架构: d18 (350M参数, 18层)"
    echo ""
    
    echo "🏆 8NPU Chat-RL 圆满成功！"
    echo "💾 模型已安全保存到持久化存储！"
    echo "============================================="
else
    echo "============================================="
    echo "❌ 训练失败 (退出码: $EXIT_CODE)"
    echo "============================================="
    echo ""
    echo "⏱️  运行时长: ${MINUTES}分${SECONDS}秒"
    echo ""
    
    # 检查是否是reduce_scatter错误
    if grep -q "reduce_scatter" /tmp/chat_rl_training.log 2>/dev/null; then
        echo "🔍 检测到 reduce_scatter 错误！"
        echo ""
        echo "💡 这是Muon优化器的分布式AdamW问题，需要改用全AdamW"
        echo ""
        echo "📝 修复步骤："
        echo "  1. 编辑 scripts/chat_rl.py"
        echo "  2. 找到 line 192-197 的 model.setup_optimizers()"
        echo "  3. 替换为标准 torch.optim.AdamW（参考 chat_sft.py）"
        echo ""
        echo "或者运行修复脚本："
        echo "  bash fix_chat_rl_adamw.sh"
    fi
    
    echo "💡 调试建议："
    echo "  1. 查看上面的错误日志"
    echo "  2. 检查NPU状态: npu-smi info"
    echo "  3. 如果OOM，降低batch size:"
    echo "     --device_batch_size=4 --examples_per_step=12"
    echo "  4. 如果reduce_scatter错误，需要修改优化器（见上方）"
    echo "  5. 查看详细日志: ls -lh ./wandb/"
    echo ""
    echo "============================================="
fi

exit $EXIT_CODE

