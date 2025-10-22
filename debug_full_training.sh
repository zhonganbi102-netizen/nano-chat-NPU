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
python3 train.py \
    --input_bin data/fineweb10B/fineweb_train_*.bin \
    --input_val_bin data/fineweb10B/fineweb_val_*.bin \
    --model d12 \
    --batch_size 4 \
    --device_batch_size 4 \
    --sequence_length 1024 \
    --num_iterations 5 \
    --learning_rate 0.0006 \
    --warmup_iters 0 \
    --warmdown_iters 0 \
    --weight_decay 0.1 \
    --val_loss_every 5 \
    --val_max_steps 5 \
    --overfit_single_batch \
    --compile False \
    --flash False \
    --tensorcores False \
    --write_tensors False \
    --inference_only False \
    --skip_bad_checkpoint True \
    --resume 0 \
    --hellaswag_eval False \
    --zero_stage 0 \
    --torch_compile False

echo "调试训练完成"