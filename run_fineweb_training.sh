#!/bin/bash

# FineWeb训练启动脚本 - 基于成功的4NPU配置
# 结合之前成功经验的精确训练命令

set -e

echo "=== FineWeb大规模训练启动 ==="

# 1. 环境准备
echo "1. 环境准备..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查数据
data_files=$(ls base_data/shard_*.parquet 2>/dev/null | wc -l || echo "0")
if [ "$data_files" -lt 50 ]; then
    echo "❌ 数据文件不足($data_files个)，请先下载数据"
    exit 1
fi
echo "✅ 检测到 $data_files 个数据文件"

# 2. 清理环境
echo "2. 清理环境..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
    ./emergency_npu_cleanup.sh
    sleep 5
fi

# 3. 设置4NPU环境变量 (基于成功配置)
echo "3. 设置4NPU环境变量..."
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29520
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 4. 创建成功验证过的优化器补丁
echo "4. 创建优化器补丁..."
cat > temp_fineweb_train_patch.py << EOF
import torch
from nanochat.gpt import GPT

def fineweb_train_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("FineWeb训练优化器设置: 全部AdamW (基于成功配置)")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    if embedding_params:
        opts.append(torch.optim.AdamW([{'params': embedding_params, 'lr': embedding_lr*0.8, 'initial_lr': embedding_lr*0.8}], lr=embedding_lr*0.8, weight_decay=weight_decay, betas=(0.9, 0.95)))
    if other_params:
        opts.append(torch.optim.AdamW([{'params': other_params, 'lr': matrix_lr*0.8, 'initial_lr': matrix_lr*0.8}], lr=matrix_lr*0.8, weight_decay=0.0, betas=(0.9, 0.95)))
    return opts

GPT.setup_optimizers = fineweb_train_optimizers
print("✅ FineWeb训练优化器补丁已应用")
EOF

# 5. 训练tokenizer (如果需要)
echo "5. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    python -m scripts.tok_train
else
    echo "tokenizer已存在，跳过训练"
fi

# 6. 开始大规模base训练
echo "6. 开始FineWeb大规模base训练..."
echo ""
echo "训练配置:"
echo "  - 数据文件: $data_files 个"
echo "  - 模型深度: 12层"
echo "  - 4NPU分布式训练"
echo "  - 批次大小: 每设备16, 总共262144"
echo "  - 训练步数: 5000"
echo "  - 预计时间: 2-4小时"
echo ""

python -c "import temp_fineweb_train_patch" && \
torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29520 \
    scripts/base_train.py \
    --model_tag=fineweb_base_d12 \
    --depth=12 \
    --device_batch_size=16 \
    --total_batch_size=262144 \
    --num_iterations=5000 \
    --embedding_lr=0.2 \
    --unembedding_lr=0.004 \
    --matrix_lr=0.02 \
    --grad_clip=1.0 \
    --eval_every=250 \
    --sample_every=1000 \
    --core_metric_every=999999

# 7. 清理
rm -f temp_fineweb_train_patch.py

echo ""
echo "🎉 FineWeb大规模训练完成！"
echo ""
echo "模型保存位置: ~/.cache/nanochat/base_checkpoints/fineweb_base_d12/"
echo ""
echo "下一步:"
echo "  - 测试模型: python -m scripts.chat_cli"
echo "  - 启动Web: python -m scripts.chat_web"
