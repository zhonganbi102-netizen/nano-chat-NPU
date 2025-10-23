#!/bin/bash

# 终极4NPU训练解决方案 - 强力环境传递
# 使用wrapper脚本确保torchrun子进程正确继承环境

set -e

echo "💪 终极4NPU FineWeb训练 - 强力环境解决方案 💪"

# 1. 强力清理
echo "1. 强力清理NPU环境..."
./emergency_npu_cleanup.sh
sleep 20

# 2. 创建环境wrapper脚本
echo "2. 创建环境wrapper脚本..."
cat > npu_env_wrapper.sh << 'EOF'
#!/bin/bash

# NPU环境wrapper - 确保所有子进程正确继承环境

# 设置基本环境变量
export ASCEND_HOME="/usr/local/Ascend/ascend-toolkit"

# 查找正确的set_env.sh路径
POSSIBLE_SET_ENV=(
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    "/usr/local/Ascend/set_env.sh"
)

for env_path in "${POSSIBLE_SET_ENV[@]}"; do
    if [ -f "$env_path" ]; then
        echo "✅ 使用环境文件: $env_path"
        source "$env_path"
        export ASCEND_HOME="$(dirname "$env_path")"
        break
    fi
done

# 强制设置所有必要的环境变量
export PATH="/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/python/site-packages:$PYTHONPATH"
export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export PYTHONPATH=".:$PYTHONPATH"

# NPU特定环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29525
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:64
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 执行传入的命令
exec "$@"
EOF

chmod +x npu_env_wrapper.sh

# 3. 创建训练脚本wrapper
echo "3. 创建训练脚本wrapper..."
cat > wrapped_base_train.py << 'EOF'
#!/usr/bin/env python3

import os
import sys

# 强制设置环境变量 (Python级别)
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'max_split_size_mb:64')

# 添加必要路径到sys.path
sys.path.insert(0, '.')
sys.path.insert(0, '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
sys.path.insert(0, '/usr/local/Ascend/ascend-toolkit/python/site-packages')

print(f"✅ Python环境wrapper: PID={os.getpid()}, RANK={os.environ.get('LOCAL_RANK', 'N/A')}")

# 导入并执行原始训练脚本
import scripts.base_train
EOF

chmod +x wrapped_base_train.py

# 4. 4NPU优化器补丁
echo "4. 创建4NPU优化器补丁..."
cat > temp_ultimate_patch.py << 'EOF'
import torch
from nanochat.gpt import GPT

def ultimate_4npu_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("💪 终极4NPU FineWeb优化器: 强力环境配置")
    
    # 获取参数分组
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    
    opts = []
    
    # 嵌入层优化器
    if embedding_params:
        embedding_opt = torch.optim.AdamW(
            [{'params': embedding_params, 'lr': embedding_lr*0.3, 'initial_lr': embedding_lr*0.3}], 
            lr=embedding_lr*0.3, 
            weight_decay=weight_decay, 
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False,
            amsgrad=False
        )
        opts.append(embedding_opt)
        print(f"  ✅ 嵌入层优化器: lr={embedding_lr*0.3:.6f}, {len(embedding_params)}个参数")
    
    # 其他参数优化器
    if other_params:
        other_opt = torch.optim.AdamW(
            [{'params': other_params, 'lr': matrix_lr*0.3, 'initial_lr': matrix_lr*0.3}], 
            lr=matrix_lr*0.3, 
            weight_decay=0.0, 
            betas=(0.9, 0.95),
            eps=1e-8,
            foreach=False,
            fused=False,
            amsgrad=False
        )
        opts.append(other_opt)
        print(f"  ✅ 其他参数优化器: lr={matrix_lr*0.3:.6f}, {len(other_params)}个参数")
    
    print(f"  ✅ 总共 {len(opts)} 个优化器")
    return opts

# 应用补丁
GPT.setup_optimizers = ultimate_4npu_optimizers
print("✅ 终极4NPU FineWeb优化器补丁已应用")
EOF

# 5. 训练tokenizer
echo "5. 训练tokenizer..."
if [ ! -f ~/.cache/nanochat/tokenizer/tokenizer.pkl ]; then
    echo "训练tokenizer..."
    ./npu_env_wrapper.sh python -m scripts.tok_train
else
    echo "✅ tokenizer已存在"
fi

# 6. 启动终极4NPU训练
echo ""
echo "🚀 启动终极4NPU FineWeb训练..."
echo ""
echo "📊 训练配置:"
echo "  - 终极4NPU配置 + wrapper环境"
echo "  - 模型深度: 8层"
echo "  - 批次大小: 每设备2, 总32768"
echo "  - 训练步数: 800步"
echo "  - 环境: 多层wrapper保护"
echo "  - 预计时间: 25-45分钟"
echo ""

# 使用wrapper启动训练
./npu_env_wrapper.sh python -c "import temp_ultimate_patch" && \
./npu_env_wrapper.sh torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29525 \
    wrapped_base_train.py \
    --model_tag=ultimate_4npu_fineweb_d8 \
    --depth=8 \
    --device_batch_size=2 \
    --total_batch_size=32768 \
    --num_iterations=800 \
    --embedding_lr=0.003 \
    --unembedding_lr=0.0003 \
    --matrix_lr=0.0015 \
    --grad_clip=0.4 \
    --eval_every=100 \
    --sample_every=400 \
    --core_metric_every=999999

# 7. 清理
rm -f temp_ultimate_patch.py npu_env_wrapper.sh wrapped_base_train.py

echo ""
echo "🎉 终极4NPU FineWeb训练完成！"
echo ""
echo "📍 模型位置: ~/.cache/nanochat/base_checkpoints/ultimate_4npu_fineweb_d8/"
echo ""
echo "💪 终极解决方案成功！"
