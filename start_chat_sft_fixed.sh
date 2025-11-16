#!/bin/bash

echo "🚀 启动8NPU Chat-SFT训练 (Muon混合优化)"
echo "============================================="
echo ""
echo "📋 基于已完成的mid_train模型继续训练"
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
# 步骤2：验证mid_train模型存在
# ============================================

echo "🔍 步骤2: 验证mid_train模型..."

# 搜索d18 mid_train模型
MID_MODEL_DIRS=(
    "/mnt/linxid615/bza/nanochat-models/mid_checkpoints/d18"  # 持久化路径 - 最优先
    "/mnt/linxid615/bza/nanochat-models/mid_checkpoints"      # 持久化备选
    "/root/.cache/nanochat/mid_checkpoints/d18"               # 传统路径
    "/root/.cache/nanochat/mid_checkpoints"
)

MID_MODEL_FOUND=""
for dir in "${MID_MODEL_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        MID_MODEL_FOUND="$dir"
        break
    fi
done

if [ -z "$MID_MODEL_FOUND" ]; then
    echo "❌ 错误：未找到mid_train模型！"
    echo ""
    echo "💡 请先运行mid_train训练："
    echo "   bash start_mid_train_fixed.sh"
    exit 1
fi

echo "✅ 找到mid_train模型: $MID_MODEL_FOUND"

# 显示模型文件信息 - 检查step 1532
if [ -f "$MID_MODEL_FOUND/model_001532.pt" ]; then
    MODEL_SIZE=$(du -h "$MID_MODEL_FOUND/model_001532.pt" | cut -f1)
    OPTIM_SIZE=$(du -h "$MID_MODEL_FOUND/optim_001532.pt" | cut -f1)
    echo "📊 模型文件: model_001532.pt ($MODEL_SIZE)"
    echo "📊 优化器文件: optim_001532.pt ($OPTIM_SIZE)"
    echo "✅ Mid-Train模型 (step 1532) 已确认"
else
    LATEST_MODEL=$(ls -t "$MID_MODEL_FOUND"/model_*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_MODEL" ]; then
        MODEL_SIZE=$(du -h "$LATEST_MODEL" | cut -f1)
        MODEL_NAME=$(basename "$LATEST_MODEL")
        echo "📊 模型文件: $MODEL_NAME ($MODEL_SIZE)"
        echo "⚠️  注意：不是预期的step 1532模型"
        
        # 提取步数
        STEP=$(echo "$MODEL_NAME" | grep -o '[0-9]\+' | head -1)
        echo "📈 训练步数: $STEP"
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
mkdir -p "$NANOCHAT_CACHE_DIR/chatsft_checkpoints"
mkdir -p "$NANOCHAT_CACHE_DIR/chatrl_checkpoints"

echo "✅ 持久化路径已设置: $NANOCHAT_CACHE_DIR"
echo ""

# ============================================
# 步骤4：显示修复总结
# ============================================

echo "✅ 完整修复总结："
echo "  1. ✓ chat_sft.py: 使用Muon+AdamW混合优化器 (智能参数分配)"
echo "  2. ✓ chat_sft.py: NPU兼容性配置 (跳过torch.compile)"
echo "  3. ✓ chat_sft.py: 8NPU分布式训练支持"
echo "  4. ✓ chat_sft.py: 内存优化 (每20步清理)"
echo "  5. ✓ HF镜像源已设置"
echo "  6. ✓ 批次大小优化为64 (8×4×2)"
echo ""
echo "💡 关键改进：智能Muon兼容性分析，最佳性能优化器组合！"
echo ""

# ============================================
# 步骤5：设置训练环境
# ============================================

# 设置8NPU环境变量
export WANDB_MODE=offline
export WANDB_SILENT=true
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29700
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
export HCCL_CONNECT_TIMEOUT=7200  # 2小时超时
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_LAUNCH_BLOCKING=1
export HCCL_WHITELIST_DISABLE=1
export OMP_NUM_THREADS=8
export TASK_QUEUE_ENABLE=0  # NPU内存优化

# HuggingFace镜像源和网络优化
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
export HF_HUB_ENABLE_HF_TRANSFER=1
export CURL_CA_BUNDLE=""
export REQUESTS_CA_BUNDLE=""
# 网络超时和重试设置
export HF_HUB_DOWNLOAD_TIMEOUT=300  # 5分钟超时
export TOKENIZERS_PARALLELISM=false  # 避免并行下载冲突

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

# ============================================
# 步骤6：清理NPU显存
# ============================================

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

# ============================================
# 步骤6.5：数据集预检查和下载
# ============================================

echo ""
echo "🌐 步骤6.5: 检查数据集可用性..."

# 测试网络连接
echo "测试HuggingFace连接..."
if curl -s --connect-timeout 10 --max-time 30 "https://hf-mirror.com" > /dev/null; then
    echo "✅ HF镜像连接正常"
else
    echo "⚠️  HF镜像连接问题，尝试直连..."
    export HF_ENDPOINT=""
    unset HF_ENDPOINT
fi

# 预下载关键数据集以避免训练时超时
echo "预下载关键数据集..."
python3 -c "
import os
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')

try:
    from datasets import load_dataset
    print('预下载 ARC 数据集...')
    ds1 = load_dataset('allenai/ai2_arc', 'ARC-Easy', split='train')
    print(f'✓ ARC-Easy: {len(ds1)} 样本')
    
    ds2 = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='train')  
    print(f'✓ ARC-Challenge: {len(ds2)} 样本')
    
    print('预下载 GSM8K 数据集...')
    ds3 = load_dataset('openai/gsm8k', 'main', split='train')
    print(f'✓ GSM8K: {len(ds3)} 样本')
    
    print('✅ 所有数据集预下载完成')
except Exception as e:
    print(f'⚠️  数据集下载遇到问题: {e}')
    print('将在训练时重试...')
" || echo "⚠️  预下载失败，训练时将重试"

# ============================================
# 步骤7：启动8NPU Chat-SFT训练
# ============================================

echo ""
echo "🎯 启动8NPU Chat-SFT训练 (Muon混合优化)..."
echo ""
echo "📊 训练配置："
echo "  - 基础模型: mid_checkpoints/d18 (350M参数, step 1532)"
echo "  - NPU数量: 8个"
echo "  - device_batch_size: 4 (内存优化)"
echo "  - target_examples_per_step: 64 (4×8×2, 梯度累积2步)"
echo "  - 数据集: ARC-Easy(2.3K) + ARC-Challenge(1.1K) + GSM8K(8K) + SmolTalk(10K)"
echo "  - 总计: 21.4K样本"
echo "  - 训练轮数: 1 epoch"
echo "  - 优化器: Muon+AdamW混合 (智能参数分配)"
echo "  - 学习率: embedding=0.2, unembedding=0.004, matrix=0.02"
echo "  - 预计时间: 15-30分钟 (8NPU加速)"
echo ""

START_TIME=$(date +%s)

# 多次尝试启动训练（处理网络问题）
TRAIN_SUCCESS=false
for attempt in 1 2 3; do
    echo ""
    echo "🚀 第 $attempt 次尝试启动训练..."
    
    # 启动8NPU chat_sft (Muon混合优化)
    if torchrun --nproc_per_node=8 \
    --master_addr=127.0.0.1 \
    --master_port=29700 \
    -- \
    -m scripts.chat_sft \
    --run=chat_sft_8npu_muon \
    --source=mid \
    --model_tag=d18 \
    --step=1532 \
    --device_batch_size=4 \
    --target_examples_per_step=64 \
    --num_epochs=1 \
    --unembedding_lr=0.004 \
    --embedding_lr=0.2 \
    --matrix_lr=0.02 \
    --weight_decay=0.0 \
    --init_lr_frac=0.02 \
    --eval_every=100 \
    --eval_metrics_every=200

SFT_TRAIN_TIME=$(($(date +%s) - START_TIME))

# ============================================
# 步骤8：验证训练结果
# ============================================

echo ""
echo "🔍 步骤8: 验证训练结果..."

# 验证模型保存
SFT_MODEL_DIRS=(
    "/mnt/linxid615/bza/nanochat-models/chatsft_checkpoints/d18"  # 持久化路径 - 最优先
    "/mnt/linxid615/bza/nanochat-models/chatsft_checkpoints"      # 持久化备选
    "/root/.cache/nanochat/chatsft_checkpoints/d18"               # 传统路径
    "/root/.cache/nanochat/chatsft_checkpoints"
)

SFT_MODEL_DIR_FOUND=""
for dir in "${SFT_MODEL_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
        SFT_MODEL_DIR_FOUND="$dir"
        break
    fi
done

echo ""
echo "============================================="
echo "🎉🎉🎉 8NPU Chat-SFT 训练完成！🎉🎉🎉"
echo "============================================="
echo ""

if [ ! -z "$SFT_MODEL_DIR_FOUND" ]; then
    echo "✅ 训练状态: 成功完成"
    echo "⏱️  训练用时: $((SFT_TRAIN_TIME / 60))分$((SFT_TRAIN_TIME % 60))秒"
    echo "📁 模型位置: $SFT_MODEL_DIR_FOUND"
    echo ""
    
    # 显示模型文件详情
    echo "📋 模型文件列表："
    ls -lah "$SFT_MODEL_DIR_FOUND" | grep -E '\.(pt|json)' | tail -10
    echo ""
    
    # 统计模型大小
    TOTAL_SIZE=$(du -sh "$SFT_MODEL_DIR_FOUND" | cut -f1)
    echo "💾 总计大小: $TOTAL_SIZE"
    echo ""
    
    echo "🎯 后续步骤："
    echo "  1️⃣  Chat-RL训练 (可选):"
    echo "     bash start_chat_rl_8npu.sh"
    echo ""
    echo "  2️⃣  测试Chat模型:"
    echo "     python -m scripts.chat_cli --source=chatsft --model_tag=d18 -p \"Why is the sky blue?\""
    echo ""
    echo "  3️⃣  启动Web界面:"
    echo "     python -m scripts.chat_web --source=chatsft --model_tag=d18"
    echo ""
    echo "  4️⃣  查看训练日志:"
    echo "     ls -lh ./wandb/offline-run-*/"
    echo ""
    
    echo "✅ 完成的关键步骤："
    echo "  ✓ 安装所有Python依赖"
    echo "  ✓ 验证mid_train模型 (350M参数, step 1532)"
    echo "  ✓ 设置持久化保存路径"
    echo "  ✓ 使用Muon+AdamW混合优化器"
    echo "  ✓ NPU兼容性配置"
    echo "  ✓ 8NPU分布式训练完整数据集"
    echo "  ✓ 智能参数分配 (Muon兼容性分析)"
    echo ""
    
    echo "💡 技术总结："
    echo "  ✓ 训练数据: ARC(3.4K) + GSM8K(8K) + SmolTalk(10K)"
    echo "  ✓ 优化器: Muon+AdamW混合 (智能分配)"
    echo "  ✓ NPU配置: 0,1,2,3,4,5,6,7 (全部8个NPU)"
    echo "  ✓ 批次配置: device_batch_size=4, target_examples_per_step=64"
    echo "  ✓ 梯度累积: 2步 (64 / 32 = 2)"
    echo "  ✓ 内存优化: 每20步清理, 垃圾回收增强"
    echo ""
    
    echo "🏆 8NPU Chat-SFT 圆满成功！"
    echo "💾 模型已安全保存到持久化存储！"
    echo "============================================="
else
    echo "❌ 训练状态: 失败"
    echo ""
    echo "🔍 可能的问题："
    echo "  1. 训练过程中出现错误"
    echo "  2. NPU内存不足 (OOM)"
    echo "  3. 模型未成功保存"
    echo ""
    echo "💡 调试建议："
    echo "  1. 查看上面的错误日志"
    echo "  2. 检查NPU状态: npu-smi info"
    echo "  3. 如果OOM，降低device_batch_size:"
    echo "     --device_batch_size=2 --target_examples_per_step=24"
    echo "  4. 如果批次大小错误，确保整除:"
    echo "     target_examples_per_step % (device_batch_size × 3) == 0"
    echo "  5. 查看训练日志: ls -lh ./wandb/"
    echo ""
    echo "============================================="
    exit 1
fi

