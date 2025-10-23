#!/bin/bash

# FineWeb数据集快速测试脚本
# 用于验证数据和环境，快速测试训练流程

set -e

echo "=== FineWeb快速测试脚本 ==="

# 检查数据
if [ ! -d "./base_data" ] || [ $(ls ./base_data/shard_*.parquet 2>/dev/null | wc -l) -lt 10 ]; then
    echo "❌ 需要先下载数据："
    echo "   ./download_fineweb_data.sh"
    exit 1
fi

# 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 清理环境
if [ -f "./emergency_npu_cleanup.sh" ]; then
    ./emergency_npu_cleanup.sh
    sleep 3
fi

# 设置4NPU环境
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29511
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32

echo "快速测试配置："
echo "  - 模型深度: 8层 (较小)"
echo "  - 设备批次: 8"
echo "  - 训练步数: 100 (快速验证)"

# 创建测试补丁
cat > temp_test_patch.py << EOF
import torch
from nanochat.gpt import GPT

def test_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("FineWeb测试优化器: AdamW")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    if embedding_params:
        opts.append(torch.optim.AdamW([{'params': embedding_params, 'lr': embedding_lr*0.8, 'initial_lr': embedding_lr*0.8}], lr=embedding_lr*0.8, weight_decay=weight_decay, betas=(0.9, 0.95)))
    if other_params:
        opts.append(torch.optim.AdamW([{'params': other_params, 'lr': matrix_lr*0.8, 'initial_lr': matrix_lr*0.8}], lr=matrix_lr*0.8, weight_decay=0.0, betas=(0.9, 0.95)))
    return opts

GPT.setup_optimizers = test_optimizers
print("✅ 测试优化器补丁已应用")
EOF

# 训练tokenizer (如果需要)
echo "训练tokenizer..."
python -m scripts.tok_train

# 快速base训练测试
echo "开始快速base训练测试..."
python -c "import temp_test_patch" && \
torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29511 \
    scripts/base_train.py \
    --run=fineweb_test_d8 \
    --depth=8 \
    --device_batch_size=8 \
    --total_batch_size=131072 \
    --num_iterations=100 \
    --embedding_lr=0.1 \
    --unembedding_lr=0.002 \
    --matrix_lr=0.01 \
    --grad_clip=0.5 \
    --eval_every=25 \
    --sample_every=50 \
    --core_metric_every=999999

# 清理
rm -f temp_test_patch.py

echo ""
echo "✅ FineWeb快速测试完成！"
echo "如果测试成功，可以运行完整训练："
echo "   ./train_with_fineweb.sh"
