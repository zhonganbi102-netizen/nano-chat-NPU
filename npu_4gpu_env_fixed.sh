#!/bin/bash

# 4NPU训练脚本 - 确保子进程环境正确
# 解决torchrun子进程环境变量继承问题

set -e

echo "=== 4NPU训练 - 环境变量修复版 ==="

# 1. 清理环境
if [ -f "./emergency_npu_cleanup.sh" ]; then
  bash ./emergency_npu_cleanup.sh
  sleep 5
fi

# 2. 显式设置和导出所有昇腾环境变量
echo "2. 设置并导出昇腾环境变量..."
export ASCEND_HOME=/usr/local/Ascend
source $ASCEND_HOME/ascend-toolkit/set_env.sh

# 3. 显式导出关键环境变量确保子进程继承
echo "3. 导出关键环境变量..."
export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH"

# 4. 设置分布式环境变量
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29507
export TORCH_COMPILE_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 5. 验证当前环境
echo "5. 验证当前环境..."
python3 -c "
import sys
print('当前PYTHONPATH:', sys.path[:3])
try:
    import tbe
    print('✅ TBE模块在主进程中可用')
except ImportError as e:
    print(f'❌ TBE模块错误: {e}')
    exit(1)
"

# 6. 创建环境设置补丁，确保每个子进程都能正确设置环境
cat > temp_env_patch.py << EOF
import os
import sys

# 显式设置Python路径，确保TBE模块可用
ascend_python_path = '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages'
tbe_path = '/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe'

if ascend_python_path not in sys.path:
    sys.path.insert(0, ascend_python_path)
if tbe_path not in sys.path:
    sys.path.insert(0, tbe_path)

# 设置环境变量
os.environ['ASCEND_HOME'] = '/usr/local/Ascend'
os.environ['PYTHONPATH'] = f"{ascend_python_path}:{tbe_path}:" + os.environ.get('PYTHONPATH', '')

# 验证TBE模块
try:
    import tbe
    print(f"✅ 进程 {os.getenv('RANK', 'main')} TBE模块加载成功")
except ImportError as e:
    print(f"❌ 进程 {os.getenv('RANK', 'main')} TBE模块加载失败: {e}")
    # 不退出，让用户看到错误但继续尝试

# 优化器补丁
import torch
from nanochat.gpt import GPT

def env_fixed_optimizers(self, unembedding_lr=0.001, embedding_lr=0.01, matrix_lr=0.01, weight_decay=0.0):
    print("环境修复版NPU优化器设置: 全部AdamW")
    embedding_params = [p for n, p in self.named_parameters() if 'emb_tok' in n]
    other_params = [p for n, p in self.named_parameters() if 'emb_tok' not in n]
    opts = []
    if embedding_params:
        opts.append(torch.optim.AdamW([{'params': embedding_params, 'lr': embedding_lr*0.8, 'initial_lr': embedding_lr*0.8}], lr=embedding_lr*0.8, weight_decay=weight_decay, betas=(0.9, 0.95)))
    if other_params:
        opts.append(torch.optim.AdamW([{'params': other_params, 'lr': matrix_lr*0.8, 'initial_lr': matrix_lr*0.8}], lr=matrix_lr*0.8, weight_decay=0.0, betas=(0.9, 0.95)))
    return opts

GPT.setup_optimizers = env_fixed_optimizers
print("✅ 环境修复版优化器补丁已应用")
EOF

# 7. 启动训练 - 使用显式环境变量
echo "7. 启动环境修复版4NPU训练..."
PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH" \
ASCEND_HOME="/usr/local/Ascend" \
python -c "import temp_env_patch" && \
PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH" \
ASCEND_HOME="/usr/local/Ascend" \
torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port=29507 \
    scripts/base_train.py \
    --model_tag=env_fixed_4npu \
    --depth=6 \
    --device_batch_size=2 \
    --total_batch_size=32768 \
    --num_iterations=500 \
    --embedding_lr=0.005 \
    --unembedding_lr=0.0005 \
    --matrix_lr=0.0025 \
    --grad_clip=0.5 \
    --eval_every=50 \
    --sample_every=250

# 8. 清理
rm -f temp_env_patch.py

echo "环境修复版4NPU训练完成！"
