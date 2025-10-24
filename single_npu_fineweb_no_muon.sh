#!/bin/bash

# 单NPU FineWeb训练 - 不使用Muon优化器
# Single NPU FineWeb training - without Muon optimizer

set -e

echo "🚀 单NPU FineWeb训练 - AdamW优化器版本"
echo "=================================================="

# 1. 环境设置
echo "1. 设置NPU环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单NPU环境变量
export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29800

# NPU内存优化
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export NPU_COMPILE_DISABLE=1
export TORCH_NPU_DISABLE_LAZY_INIT=1

# Python优化
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "✅ 环境变量设置完成"
echo "  设备: NPU:0"
echo "  内存优化: 128MB分割"
echo "  编译优化: 禁用"

# 2. 清理NPU内存
echo ""
echo "2. 清理NPU内存..."
python3 -c "
import torch
import torch_npu
import gc
import time

if torch_npu.npu.is_available():
    print(f'清理 {torch_npu.npu.device_count()} 个NPU设备...')
    for i in range(torch_npu.npu.device_count()):
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    
    time.sleep(1)
    gc.collect()
    print('✅ NPU内存清理完成')
"

# 3. 检查数据
echo ""
echo "3. 检查FineWeb数据..."
data_files=$(find . -name "*.parquet" 2>/dev/null | wc -l)
echo "找到 $data_files 个parquet文件"

if [ "$data_files" -lt 5 ]; then
    echo "⚠️  数据文件较少，但继续训练..."
fi

# 4. 检查tokenizer
echo ""
echo "4. 检查tokenizer..."
if [ ! -f "tokenizer/tokenizer.json" ]; then
    echo "创建简单tokenizer..."
    mkdir -p tokenizer
    python3 -c "
import json
tokenizer_config = {
    'version': '1.0',
    'model': {'type': 'BPE', 'vocab': {'<unk>': 0, '<s>': 1, '</s>': 2}, 'merges': []},
    'pre_tokenizer': {'type': 'Whitespace'},
    'post_processor': {'type': 'TemplateProcessing'}
}
with open('tokenizer/tokenizer.json', 'w') as f:
    json.dump(tokenizer_config, f)
print('✅ 简单tokenizer已创建')
"
else
    echo "✅ tokenizer已存在"
fi

# 5. 开始单NPU训练（不使用Muon）
echo ""
echo "5. 开始单NPU FineWeb训练..."
echo "⚠️  使用AdamW优化器，避免Muon兼容性问题"
echo ""

# 创建训练脚本
cat > temp_single_npu_train.py << 'EOF'
import os
import sys
import torch
import torch_npu

# 设置NPU设备
if torch_npu.npu.is_available():
    torch_npu.npu.set_device(0)
    print(f"使用设备: npu:{torch_npu.npu.current_device()}")

# 导入训练模块
sys.path.append('.')

if __name__ == "__main__":
    print("🚀 启动单NPU FineWeb训练（无Muon）...")
    
    # 修改命令行参数，强制使用AdamW
    sys.argv = [
        'base_train.py',
        '--run=single_npu_fineweb_no_muon',
        '--depth=6',                    # 中等深度
        '--device_batch_size=4',        # 适中batch size
        '--total_batch_size=8192',      # 总batch size
        '--num_iterations=1000',        # 测试用迭代次数
        '--embedding_lr=0.001',
        '--unembedding_lr=0.0001',
        '--matrix_lr=0.0005',
        '--grad_clip=1.0',
        '--eval_every=100',
        '--sample_every=500',
        '--core_metric_every=999999',
        '--optimizer=adamw',            # 强制使用AdamW
        '--verbose'
    ]
    
    try:
        from scripts.base_train import main
        main()
        print("✅ 训练成功完成！")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF

# 执行训练
python3 temp_single_npu_train.py

# 清理临时文件
rm -f temp_single_npu_train.py

echo ""
echo "🎉 单NPU FineWeb训练完成！"
echo ""
echo "如果成功，可以尝试："
echo "  1. 增加深度: depth=8 或 depth=12"
echo "  2. 增加batch size: device_batch_size=8"
echo "  3. 增加迭代次数: num_iterations=4000"