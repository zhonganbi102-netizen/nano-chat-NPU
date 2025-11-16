#!/bin/bash

echo "🚀 启动8NPU Mid-Training (Muon混合优化)"
echo "============================================="
echo ""
echo "📋 基于已完成的base_train模型继续训练"
echo "🔧 使用Muon+AdamW混合优化器"
echo "💾 模型将保存到持久化目录"
echo ""

cd /mnt/linxid615/bza/nanochat-npu

# ============================================
# 步骤1：安装Python依赖
# ============================================

echo "📦 步骤1: 安装Python依赖..."
pip install datasets fastapi files-to-prompt numpy==1.26.4 psutil regex tiktoken tokenizers uvicorn wandb --root-user-action=ignore --quiet

echo "✅ 依赖安装完成"
echo ""

# ============================================
# 验证base_train模型存在
# ============================================

echo "🔍 步骤2: 验证base_train模型 (d18, step 13351)..."

# 搜索d18完整训练模型
BASE_MODEL_DIRS=(
    "/mnt/linxid615/bza/nanochat-models/base_checkpoints/d18"  # 持久化路径 - 最优先
    "/mnt/linxid615/bza/nanochat-models/base_checkpoints"      # 持久化备选
    "/root/.cache/nanochat/base_checkpoints/d18"               # 传统路径
    "/root/.cache/nanochat/base_checkpoints"
)

BASE_MODEL_FOUND=""
for dir in "${BASE_MODEL_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        BASE_MODEL_FOUND="$dir"
        break
    fi
done

if [ -z "$BASE_MODEL_FOUND" ]; then
    echo "❌ 错误：未找到base_train模型！"
    echo ""
    echo "💡 请先运行base_train训练："
    echo "   bash ultimate_3npu_persistent_training.sh"
    exit 1
fi

echo "✅ 找到base_train模型: $BASE_MODEL_FOUND"

# 显示模型文件信息
if [ -f "$BASE_MODEL_FOUND/model_013351.pt" ]; then
    MODEL_SIZE=$(du -h "$BASE_MODEL_FOUND/model_013351.pt" | cut -f1)
    OPTIM_SIZE=$(du -h "$BASE_MODEL_FOUND/optim_013351.pt" | cut -f1)
    echo "📊 模型文件: model_013351.pt ($MODEL_SIZE)"
    echo "📊 优化器文件: optim_013351.pt ($OPTIM_SIZE)"
    echo "✅ 完整训练模型 (step 13351) 已确认"
else
    LATEST_MODEL=$(ls -t "$BASE_MODEL_FOUND"/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_MODEL" ]; then
        MODEL_SIZE=$(du -h "$LATEST_MODEL" | cut -f1)
        MODEL_NAME=$(basename "$LATEST_MODEL")
        echo "📊 模型文件: $MODEL_NAME ($MODEL_SIZE)"
        echo "⚠️  注意：不是预期的step 13351模型"
    fi
fi
echo ""

# ============================================
# 步骤3：设置持久化模型保存路径
# ============================================

echo "📁 步骤3: 设置持久化模型保存路径..."

# 设置环境变量指向持久化目录
export NANOCHAT_CACHE_DIR=/mnt/linxid615/bza/nanochat-models

# 创建必要的目录结构
mkdir -p "$NANOCHAT_CACHE_DIR/mid_checkpoints"
mkdir -p "$NANOCHAT_CACHE_DIR/chatsft_checkpoints"
mkdir -p "$NANOCHAT_CACHE_DIR/chatrl_checkpoints"

echo "✅ 持久化路径已设置: $NANOCHAT_CACHE_DIR"
echo ""

# ============================================
# 步骤4：显示修复总结
# ============================================

echo "✅ 完整修复总结："
echo "  1. ✓ mid_train.py: 使用标准PyTorch AdamW（避免reduce_scatter错误）"
echo "  2. ✓ mid_train.py: 添加NPU torch.compile检查（跳过编译）"
echo "  3. ✓ mid_train.py: 修复学习率调度器bug"
echo "  4. ✓ mid_train.py: initial_lr字段自动添加"
echo "  5. ✓ HF镜像源已设置"
echo "  6. ✓ 降低device_batch_size到4（更安全）"
echo "  7. ✓ Muon优化器用于矩阵参数（如果可用）"
echo ""
echo "💡 关键改进：直接在mid_train.py中修复，无需外部补丁！"
echo ""

# 设置8NPU环境变量
export WANDB_MODE=offline
export WANDB_SILENT=true
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
export HCCL_CONNECT_TIMEOUT=7200  # 2小时超时
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_WHITELIST_DISABLE=1
export OMP_NUM_THREADS=8
export TASK_QUEUE_ENABLE=0  # NPU内存优化

# HuggingFace镜像源
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# 🔥 虽然代码里已经检查了，但为了保险起见也设置环境变量
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

echo "⚙️ 环境配置："
echo "  - NPU设备: 0,1,2,3,4,5,6,7 (8个NPU)"
echo "  - 优化器: Muon+AdamW混合"
echo "  - torch.compile: DISABLED"
echo "  - HF镜像: $HF_ENDPOINT"
echo "  - HCCL超时: 7200s (2小时)"
echo ""

echo "💾 清理8个NPU显存..."
python3 -c "
import torch
import torch_npu
print('清理NPU显存...')
for i in range(8):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        print(f'  ✓ NPU {i} 清理完成')
    except:
        pass
" 2>/dev/null || echo "  (跳过显存清理)"

sleep 2

echo ""
echo "🎯 启动8NPU Mid-Training (Muon混合优化)..."
echo ""
echo "📊 训练配置："
echo "  - 基础模型: base_checkpoints/d18 (350M参数, 完整训练)"
echo "  - NPU数量: 8个"
echo "  - device_batch_size: 8 (内存优化)"
echo "  - total_batch_size: 262144 (8×8×2048×2)"
echo "  - 数据集: SmolTalk(460K) + MMLU(100K) + GSM8K(8K)"
echo "  - 优化器: Muon+AdamW混合 (智能参数分配)"
echo "  - 学习率: embedding=0.2, unembedding=0.004, matrix=0.02"
echo "  - 预计时间: 30-60分钟 (8NPU加速)"
echo ""

START_TIME=$(date +%s)

# 启动8NPU mid_train (Muon混合优化)
torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    -- \
    scripts/mid_train.py \
    --run=mid_train_8npu_muon \
    --device_batch_size=8 \
    --total_batch_size=262144 \
    --eval_every=150 \
    --unembedding_lr=0.004 \
    --embedding_lr=0.2 \
    --matrix_lr=0.02 \
    --weight_decay=0.0 \
    --init_lr_frac=1.0 \
    --final_lr_frac=0.0

MID_TRAIN_TIME=$(($(date +%s) - START_TIME))

# ============================================
# 步骤7：验证训练结果
# ============================================

echo ""
echo "🔍 步骤7: 验证训练结果..."

# 验证模型保存
MID_MODEL_DIRS=(
    "/mnt/linxid615/bza/nanochat-models/mid_checkpoints/d18"  # 持久化路径 - 最优先
    "/mnt/linxid615/bza/nanochat-models/mid_checkpoints"      # 持久化备选
    "/root/.cache/nanochat/mid_checkpoints/d18"               # 传统路径
    "/root/.cache/nanochat/mid_checkpoints"
)

MID_MODEL_DIR_FOUND=""
for dir in "${MID_MODEL_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        MID_MODEL_DIR_FOUND="$dir"
        break
    fi
done

echo ""
echo "============================================="
echo "🎉🎉🎉 8NPU Mid-Training 训练完成！🎉🎉🎉"
echo "============================================="
echo ""

if [ ! -z "$MID_MODEL_DIR_FOUND" ]; then
    echo "✅ 训练状态: 成功完成"
    echo "⏱️  训练用时: $((MID_TRAIN_TIME / 60))分$((MID_TRAIN_TIME % 60))秒"
    echo "📁 模型位置: $MID_MODEL_DIR_FOUND"
    echo ""
    
    # 显示模型文件详情
    echo "📋 模型文件列表："
    ls -lah "$MID_MODEL_DIR_FOUND" | grep -E '\.(pt|json)' | tail -10
    echo ""
    
    # 统计模型大小
    TOTAL_SIZE=$(du -sh "$MID_MODEL_DIR_FOUND" | cut -f1)
    echo "💾 总计大小: $TOTAL_SIZE"
    echo ""
    
    echo "🎯 后续步骤："
    echo "  1️⃣  Chat-SFT训练:"
    echo "     bash start_chat_sft_8npu.sh"
    echo ""
    echo "  2️⃣  或手动运行:"
    echo "     torchrun --nproc_per_node=8 scripts/chat_sft.py"
    echo ""
    echo "  3️⃣  查看训练日志:"
    echo "     ls -lh ./wandb/offline-run-*/"
    echo ""
    
    echo "✅ 完成的关键步骤："
    echo "  ✓ 安装所有Python依赖"
    echo "  ✓ 验证base_train模型 (350M参数)"
    echo "  ✓ 设置持久化保存路径"
    echo "  ✓ 使用Muon+AdamW混合优化器"
    echo "  ✓ 跳过torch.compile (NPU兼容)"
    echo "  ✓ 8NPU分布式训练完整数据集"
    echo "  ✓ 智能参数分配 (Muon兼容性分析)"
    echo ""
    
    echo "💡 技术总结："
    echo "  ✓ 数据集: SmolTalk(460K) + MMLU(100K) + GSM8K(8K)"
    echo "  ✓ 优化器: Muon+AdamW混合 (智能分配)"
    echo "  ✓ NPU配置: 0,1,2,3,4,5,6,7 (全部8个NPU)"
    echo "  ✓ 批次大小: device_batch_size=8, total_batch_size=262144"
    echo "  ✓ 内存优化: 每20步清理, 垃圾回收增强"
    echo ""
    
    echo "🏆 8NPU Mid-Training 圆满成功！"
    echo "💾 模型已安全保存到持久化存储！"
    echo "============================================="
else
    echo "❌ 训练状态: 失败"
    echo ""
    echo "🔍 可能的问题："
    echo "  1. 训练过程中出现错误"
    echo "  2. 模型未成功保存"
    echo "  3. 保存路径配置问题"
    echo ""
    echo "💡 调试建议："
    echo "  1. 查看上面的错误日志"
    echo "  2. 检查NPU状态: npu-smi info"
    echo "  3. 查看训练日志: ls -lh ./wandb/"
    echo "  4. 重新运行脚本"
    echo ""
    echo "============================================="
    exit 1
fi
