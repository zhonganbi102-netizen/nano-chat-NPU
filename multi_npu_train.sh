#!/bin/bash

echo "=== 🎯 多NPU训练配置脚本 ==="

# 检查可用NPU数量
echo "检查NPU状态..."
AVAILABLE_NPUS=$(python3 -c "
try:
    import torch_npu
    print(torch_npu.npu.device_count())
except:
    print(0)
")

echo "可用NPU数量: $AVAILABLE_NPUS"

if [ "$AVAILABLE_NPUS" -lt 2 ]; then
    echo "❌ 需要至少2个NPU进行分布式训练"
    echo "当前只有 $AVAILABLE_NPUS 个NPU可用"
    exit 1
fi

echo ""
echo "选择训练配置:"
echo "1) 2NPU训练 (推荐开始)"
echo "2) 4NPU训练 (高性能)"
echo "3) 8NPU训练 (最大性能)"
echo "4) 全部可用NPU ($AVAILABLE_NPUS 个)"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        NUM_NPUS=2
        DEVICE_BATCH_SIZE=12
        TOTAL_BATCH_SIZE=262144
        ;;
    2)
        NUM_NPUS=4
        DEVICE_BATCH_SIZE=8
        TOTAL_BATCH_SIZE=262144
        ;;
    3)
        NUM_NPUS=8
        DEVICE_BATCH_SIZE=6
        TOTAL_BATCH_SIZE=524288
        ;;
    4)
        NUM_NPUS=$AVAILABLE_NPUS
        DEVICE_BATCH_SIZE=$((64 / NUM_NPUS))  # 自动调整batch size
        TOTAL_BATCH_SIZE=524288
        ;;
    *)
        echo "无效选择，使用2NPU配置"
        NUM_NPUS=2
        DEVICE_BATCH_SIZE=12
        TOTAL_BATCH_SIZE=262144
        ;;
esac

# 检查NPU数量是否足够
if [ "$NUM_NPUS" -gt "$AVAILABLE_NPUS" ]; then
    echo "❌ 选择的NPU数量 ($NUM_NPUS) 超过可用数量 ($AVAILABLE_NPUS)"
    NUM_NPUS=$AVAILABLE_NPUS
    echo "调整为使用 $NUM_NPUS 个NPU"
fi

# 模型配置选择
echo ""
echo "选择模型大小:"
echo "1) 小模型 (depth=8, ~25M参数, 快速训练)"
echo "2) 中模型 (depth=12, ~85M参数, 推荐)"
echo "3) 大模型 (depth=16, ~200M参数, 需要更多内存)"

read -p "请选择 (1-3): " model_choice

case $model_choice in
    1)
        DEPTH=8
        ;;
    2)
        DEPTH=12
        ;;
    3)
        DEPTH=16
        # 大模型需要减小batch size
        if [ "$DEPTH" -eq 16 ]; then
            DEVICE_BATCH_SIZE=$((DEVICE_BATCH_SIZE / 2))
        fi
        ;;
    *)
        echo "无效选择，使用中模型"
        DEPTH=12
        ;;
esac

# 生成NPU设备列表
NPU_DEVICES=""
for i in $(seq 0 $((NUM_NPUS-1))); do
    if [ $i -eq 0 ]; then
        NPU_DEVICES="$i"
    else
        NPU_DEVICES="$NPU_DEVICES,$i"
    fi
done

echo ""
echo "=== 训练配置确认 ==="
echo "NPU数量: $NUM_NPUS"
echo "NPU设备: $NPU_DEVICES"
echo "模型深度: $DEPTH"
echo "单设备batch size: $DEVICE_BATCH_SIZE"
echo "总batch size: $TOTAL_BATCH_SIZE"
echo "序列长度: 2048"

# 计算预期性能
TOKENS_PER_STEP=$((DEVICE_BATCH_SIZE * 2048 * NUM_NPUS))
echo "每步tokens: $TOKENS_PER_STEP"

read -p "确认开始训练? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "取消训练"
    exit 0
fi

# 设置环境变量
export ASCEND_RT_VISIBLE_DEVICES=$NPU_DEVICES
export WORLD_SIZE=$NUM_NPUS
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 清理之前的进程
echo "清理之前的训练进程..."
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 2

# 创建训练命令
TRAIN_CMD="torchrun --standalone --nproc_per_node=$NUM_NPUS scripts/base_train.py"
TRAIN_CMD="$TRAIN_CMD --depth=$DEPTH"
TRAIN_CMD="$TRAIN_CMD --device_batch_size=$DEVICE_BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --total_batch_size=$TOTAL_BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --run=\"${NUM_NPUS}npu_d${DEPTH}_$(date +%Y%m%d_%H%M%S)\""

echo ""
echo "🚀 启动 ${NUM_NPUS}NPU 分布式训练..."
echo "命令: $TRAIN_CMD"
echo ""

# 启动训练
eval $TRAIN_CMD

echo ""
echo "训练完成: $(date)"