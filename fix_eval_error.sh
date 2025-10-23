#!/bin/bash

# 修复评估文件缺失问题的脚本

echo "=== 修复评估文件缺失问题 ==="

# 禁用CORE metric评估，避免文件缺失错误
echo "创建4NPU训练脚本 - 禁用评估..."

cat > npu_4gpu_no_eval.sh << EOF
#!/bin/bash

# 4NPU训练脚本 - 禁用CORE评估避免文件缺失错误

set -e

echo "=== 4NPU训练 - 禁用评估版 ==="

# 1. 设置环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29509
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 2. 创建优化器补丁
cat > temp_no_eval_patch.py << EOL
import torch
from nanochat.gpt import GPT

def no_eval_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("4NPU优化器设置: 全部AdamW")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    if embedding_params:
        opts.append(torch.optim.AdamW([{'params': embedding_params, 'lr': embedding_lr*0.8, 'initial_lr': embedding_lr*0.8}], lr=embedding_lr*0.8, weight_decay=weight_decay, betas=(0.9, 0.95)))
    if other_params:
        opts.append(torch.optim.AdamW([{'params': other_params, 'lr': matrix_lr*0.8, 'initial_lr': matrix_lr*0.8}], lr=matrix_lr*0.8, weight_decay=0.0, betas=(0.9, 0.95)))
    return opts

GPT.setup_optimizers = no_eval_optimizers
print("✅ 4NPU优化器补丁已应用")
EOL

# 3. 启动训练 - 禁用CORE评估
python -c "import temp_no_eval_patch" && \\
torchrun --nproc_per_node=4 \\
    --master_addr=127.0.0.1 \\
    --master_port=29509 \\
    scripts/base_train.py \\
    --model_tag=npu_4gpu_no_eval \\
    --depth=6 \\
    --device_batch_size=2 \\
    --total_batch_size=32768 \\
    --num_iterations=1000 \\
    --embedding_lr=0.005 \\
    --unembedding_lr=0.0005 \\
    --matrix_lr=0.0025 \\
    --grad_clip=0.5 \\
    --eval_every=50 \\
    --sample_every=250 \\
    --core_metric_every=999999

# 4. 清理
rm -f temp_no_eval_patch.py

echo "4NPU训练完成！"
EOF

chmod +x npu_4gpu_no_eval.sh

echo "✅ 已创建 npu_4gpu_no_eval.sh"
echo "这个脚本设置 --core_metric_every=999999 来禁用CORE评估"
echo "避免缺少eval文件的错误"
