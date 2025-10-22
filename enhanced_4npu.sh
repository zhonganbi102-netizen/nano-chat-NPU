#!/bin/bash

echo "🚀 启动增强版4NPU分布式训练..."
echo "⏰ 开始时间: $(date)"

# 清理之前的进程
pkill -f "python.*base_train" || true
sleep 2

# 设置严格的NPU环境变量
export ASCEND_RT_DEBUG_LEVEL=INFO
export ASCEND_GLOBAL_LOG_LEVEL=1
export HCCL_WHITELIST_DISABLE=1

# HCCL通信优化配置
export HCCL_CONNECT_TIMEOUT=600     # 10分钟连接超时
export HCCL_EXEC_TIMEOUT=600        # 10分钟执行超时  
export HCCL_HEARTBEAT_TIMEOUT=600   # 10分钟心跳超时
export HCCL_REDUCE_OP_SYNC=1        # 同步reduce操作
export HCCL_BUFFSIZE=64              # 减小缓冲区大小

# 分布式训练配置
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12357

# 内存和性能配置
export NPU_FUZZY_COMPILE_BLACKLIST="HcclAllreduce,HcclAllgather,HcclReduceScatter"
export TASK_QUEUE_ENABLE=0

# Python调试配置
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

echo "🔧 环境配置完成"
echo "📊 NPU设备信息:"
npu-smi info | head -20

echo "🧪 运行增强版4NPU训练..."

# 训练参数
GRAD_ACCUM_STEPS=1
DEVICE_BATCH_SIZE=2
TOTAL_BATCH_SIZE=4096  # 2 * 512 * 4 = 4096 (必须能被world_tokens_per_fwdbwd整除)
SEQ_LEN=512

# 计算实际的梯度累积步数  
DDPWORLD=4
WORLD_TOKENS_PER_FWDBWD=$((DEVICE_BATCH_SIZE * SEQ_LEN * DDPWORLD))
REAL_GRAD_ACCUM=$((TOTAL_BATCH_SIZE / WORLD_TOKENS_PER_FWDBWD))

echo "📈 训练配置:"
echo "  - 设备批次大小: $DEVICE_BATCH_SIZE"
echo "  - 总批次大小: $TOTAL_BATCH_SIZE" 
echo "  - 梯度累积步数: $REAL_GRAD_ACCUM"
echo "  - DDP世界大小: $DDPWORLD"
echo "  - 序列长度: $SEQ_LEN"

# 启动训练
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/base_train.py \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --max_seq_len=$SEQ_LEN \
    --depth=20 \
    --num_iterations=500 \
    --eval_every=50 \
    --eval_tokens=10240 \
    --core_metric_every=999999

TRAIN_EXIT_CODE=$?

echo "⏰ 结束时间: $(date)"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ 增强版4NPU训练成功完成!"
else
    echo "❌ 增强版4NPU训练失败，退出码: $TRAIN_EXIT_CODE"
    
    echo "🔍 诊断信息:"
    echo "📊 NPU状态:"
    npu-smi info
    
    echo "🔗 网络状态:"
    netstat -tuln | grep 12357 || echo "端口12357未监听"
    
    echo "🏃‍♂️ 进程状态:"
    ps aux | grep -E "(python|torchrun)" | grep -v grep || echo "无相关进程"
fi

echo "🏁 增强版4NPU训练完成: $(date)"