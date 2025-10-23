#!/bin/bash

# FineWeb数据集完整训练脚本 - NPU 4卡版本
# 基于成功的4NPU配置进行大规模训练

set -e

echo "=== FineWeb数据集NPU训练管道 ==="

# 0. 环境检查和设置
echo "0. 环境检查和设置..."

# 检查数据是否存在
if [ ! -d "./base_data" ] || [ $(ls ./base_data/shard_*.parquet 2>/dev/null | wc -l) -lt 50 ]; then
    echo "❌ 数据集不存在或文件不足，请先运行数据下载脚本："
    echo "   ./download_fineweb_data.sh"
    exit 1
fi

data_files=$(ls ./base_data/shard_*.parquet 2>/dev/null | wc -l)
echo "✅ 检测到 $data_files 个数据文件"

# 设置昇腾环境
echo "设置昇腾NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查NPU状态
echo "检查NPU状态..."
npu-smi info | head -20

# 验证torch_npu
python3 -c "
import torch
import torch_npu
assert torch_npu.npu.is_available(), 'NPU不可用'
print(f'✅ NPU设备数量: {torch_npu.npu.device_count()}')
print(f'✅ torch_npu版本: {torch_npu.__version__}')
"

# 1. 清理残留进程
echo "1. 清理残留进程..."
if [ -f "./emergency_npu_cleanup.sh" ]; then
    ./emergency_npu_cleanup.sh
    sleep 5
fi

# 2. 设置4NPU环境变量
echo "2. 设置4NPU分布式环境..."
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29510
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

echo "NPU配置:"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"

# 3. 训练Tokenizer
echo "3. 训练Tokenizer..."
echo "使用FineWeb数据训练tokenizer..."
python -m scripts.tok_train

# 4. 创建优化器补丁
echo "4. 创建4NPU优化器补丁..."
cat > temp_fineweb_patch.py << EOF
import torch
from nanochat.gpt import GPT

def fineweb_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("FineWeb 4NPU优化器设置: 全部AdamW")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    if embedding_params:
        opts.append(torch.optim.AdamW([{'params': embedding_params, 'lr': embedding_lr*0.8, 'initial_lr': embedding_lr*0.8}], lr=embedding_lr*0.8, weight_decay=weight_decay, betas=(0.9, 0.95)))
    if other_params:
        opts.append(torch.optim.AdamW([{'params': other_params, 'lr': matrix_lr*0.8, 'initial_lr': matrix_lr*0.8}], lr=matrix_lr*0.8, weight_decay=0.0, betas=(0.9, 0.95)))
    return opts

GPT.setup_optimizers = fineweb_optimizers
print("✅ FineWeb 4NPU优化器补丁已应用")
EOF

# 5. Base Model训练 - 大规模版本
echo "5. 开始Base Model训练 (FineWeb数据集)..."
echo "配置："
echo "  - 模型深度: 12层"
echo "  - 批次大小: 每设备16, 总批次262144"
echo "  - 训练步数: 5000 (可根据需要调整)"
echo "  - 学习率: 优化的NPU兼容配置"

python -c "import temp_fineweb_patch" && \
torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29510 \
    scripts/base_train.py \
    --run=fineweb_base_d12 \
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

# 6. 清理临时文件
rm -f temp_fineweb_patch.py

echo ""
echo "🎉 FineWeb Base训练完成！"
echo ""
echo "模型保存位置: ~/.cache/nanochat/base_checkpoints/fineweb_base_d12/"
echo ""
echo "训练统计："
echo "  - 数据文件: $data_files 个"
echo "  - 训练步数: 5000"
echo "  - 使用NPU: 4卡"
echo ""
echo "下一步选项："
echo "1. 运行中间训练: ./run_midtraining.sh"
echo "2. 进行Chat SFT: ./run_chat_sft.sh" 
echo "3. 测试模型: python -m scripts.chat_cli"
echo "4. 启动Web服务: python -m scripts.chat_web"
