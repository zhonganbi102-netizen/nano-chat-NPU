#!/bin/bash

echo "=== 调试完整训练脚本 ==="

# 清理环境
source clean_npu_environment.sh

# 设置4NPU分布式环境（但先用单NPU测试）
export WORLD_SIZE=1  # 暂时用单NPU
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export ASCEND_RT_VISIBLE_DEVICES=0

cd /mnt/linxid615/bza/nanochat-npu

echo "1. 验证训练脚本参数..."
echo "使用单NPU调试模式，批次大小调整为最小值"

# 创建调试版本的训练命令
python3 scripts/base_train.py \
    --depth=2 \
    --max_seq_len=512 \
    --device_batch_size=4 \
    --total_batch_size=2048 \
    --num_iterations=5 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.002 \
    --matrix_lr=0.01 \
    --weight_decay=0.01 \
    --grad_clip=1.0 \
    --eval_every=5 \
    --eval_tokens=10240 \
    --core_metric_every=10 \
    --core_metric_max_per_task=10 \
    --sample_every=10 \
    --run=debug_single_npu

echo "调试训练完成"