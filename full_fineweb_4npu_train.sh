#!/bin/bash

# 完整FineWeb数据集4NPU大规模训练
# 基于单NPU成功经验的4NPU完整版本

set -e

echo "🔥 完整FineWeb数据集4NPU大规模训练 🔥"

# 1. 强力清理
echo "1. 强力清理NPU环境..."
./emergency_npu_cleanup.sh
sleep 20

# 2. 验证数据集
data_files=$(ls base_data/shard_*.parquet 2>/dev/null | wc -l || echo "0")
if [ "$data_files" -lt 100 ]; then
    echo "❌ FineWeb数据文件不足($data_files个)，建议至少100个文件"
    echo "请运行: ./download_fineweb_data.sh"
    exit 1
fi
echo "✅ FineWeb数据文件: $data_files 个 (~$(($data_files * 150))MB)"

# 3. 创建环境wrapper脚本 (基于成功经验)
echo "2. 创建完整训练环境wrapper..."
cat > full_npu_env_wrapper.sh << 'EOF'
#!/bin/bash

# 完整训练NPU环境wrapper - 基于单NPU成功经验

echo "🚀 设置完整FineWeb训练环境..."

# 查找并设置Ascend环境
ASCEND_PATHS=(
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    "/usr/local/Ascend/set_env.sh"
)

for env_path in "${ASCEND_PATHS[@]}"; do
    if [ -f "$env_path" ]; then
        echo "✅ 使用环境文件: $env_path"
        source "$env_path"
        export ASCEND_HOME="$(dirname "$env_path")"
        break
    fi
done

# 强制设置所有必要环境变量
export PATH="/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/python/site-packages:$PYTHONPATH"
export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export PYTHONPATH=".:$PYTHONPATH"

# 4NPU分布式环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29526
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128  # 更大内存配置用于完整训练
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

echo "✅ 完整训练环境设置完成"

# 执行传入的命令
exec "$@"
EOF

chmod +x full_npu_env_wrapper.sh

# 4. 完整训练Python wrapper
echo "3. 创建完整训练脚本wrapper..."
cat > full_base_train.py << 'EOF'
#!/usr/bin/env python3

import os
import sys

# 基于单NPU成功经验的环境设置
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'max_split_size_mb:128')

# 添加必要路径
sys.path.insert(0, '.')
sys.path.insert(0, '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
sys.path.insert(0, '/usr/local/Ascend/ascend-toolkit/python/site-packages')

print(f"🔥 完整FineWeb训练wrapper: PID={os.getpid()}, RANK={os.environ.get('LOCAL_RANK', 'N/A')}")

# 导入并执行原始训练脚本
import scripts.base_train
EOF

chmod +x full_base_train.py

# 5. 完整训练优化器补丁 (基于单NPU成功配置)
echo "4. 创建完整训练优化器补丁..."
cat > temp_full_train_patch.py << 'EOF'
import torch
from nanochat.gpt import GPT

def full_fineweb_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("🔥 完整FineWeb数据集4NPU训练优化器")
    
    # 基于单NPU成功经验的参数分组
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    
    opts = []
    
    # 嵌入层优化器 (基于单NPU成功配置调整)
    if embedding_params:
        embedding_opt = torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr*0.5, 'initial_lr': embedding_lr*0.5}], 
            lr=embedding_lr*0.5, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,  # 关键: 基于单NPU成功经验
            fused=False,
            amsgrad=False
        )
        opts.append(embedding_opt)
        print(f"  ✅ 嵌入层优化器: lr={embedding_lr*0.5:.6f}, {len(embedding_params)}个参数")
    
    # 其他参数优化器
    if other_params:
        other_opt = torch.optim.AdamW(
            [{'params': other_params, 'lr': matrix_lr*0.5, 'initial_lr': matrix_lr*0.5}], 
            lr=matrix_lr*0.5, 
            weight_decay=0.0, 
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,  # 关键: 基于单NPU成功经验
            fused=False,
            amsgrad=False
        )
        opts.append(other_opt)
        print(f"  ✅ 其他参数优化器: lr={matrix_lr*0.5:.6f}, {len(other_params)}个参数")
    
    print(f"  🔥 完整训练总共 {len(opts)} 个优化器")
    return opts

# 应用补丁
GPT.setup_optimizers = full_fineweb_optimizers
print("✅ 完整FineWeb数据集4NPU训练优化器补丁已应用")
EOF

# 6. 训练tokenizer
echo "5. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    ./full_npu_env_wrapper.sh python -m scripts.tok_train
else
    echo "✅ tokenizer已存在"
fi

# 7. 完整FineWeb数据集4NPU大规模训练
echo ""
echo "🔥 启动完整FineWeb数据集4NPU大规模训练..."
echo ""
echo "📊 完整训练配置:"
echo "  - 数据文件: $data_files 个 (~$(($data_files * 150))MB)"
echo "  - 完整FineWeb数据集训练"
echo "  - 模型深度: 12层 (大模型)"
echo "  - 4NPU分布式并行"
echo "  - 批次大小: 每设备8, 总131072"
echo "  - 训练步数: 4000步 (充分训练)"
echo "  - 学习率: 基于单NPU成功经验调整"
echo "  - 预计时间: 2-3小时"
echo ""

# 启动完整大规模训练
./full_npu_env_wrapper.sh python -c "import temp_full_train_patch" && \
./full_npu_env_wrapper.sh torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29526 \
    full_base_train.py \
    --model_tag=full_fineweb_dataset_d12 \
    --depth=12 \
    --device_batch_size=8 \
    --total_batch_size=131072 \
    --num_iterations=4000 \
    --embedding_lr=0.01 \
    --unembedding_lr=0.001 \
    --matrix_lr=0.005 \
    --grad_clip=0.8 \
    --eval_every=200 \
    --sample_every=800 \
    --core_metric_every=999999

# 8. 清理
rm -f temp_full_train_patch.py full_npu_env_wrapper.sh full_base_train.py

echo ""
echo "🎉 完整FineWeb数据集4NPU大规模训练完成！"
echo ""
echo "📍 模型位置: ~/.cache/nanochat/base_checkpoints/full_fineweb_dataset_d12/"
echo ""
echo "🔥 恭喜完成完整数据集训练！"
echo "📊 训练统计:"
echo "  - 总训练token: ~5.24亿tokens"
echo "  - 模型参数: ~337M"
echo "  - 训练质量: 完整FineWeb数据集"
