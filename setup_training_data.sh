#!/bin/bash

echo "=== 数据下载和验证工具 ==="

cd /mnt/linxid615/bza/nanochat-npu

echo "1. 检查当前数据目录状态..."

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

from nanochat.common import get_base_dir
from nanochat.dataset import DATA_DIR, list_parquet_files

print(f'Base目录: {get_base_dir()}')
print(f'数据目录: {DATA_DIR}')
print(f'数据目录存在: {os.path.exists(DATA_DIR)}')

if os.path.exists(DATA_DIR):
    files = os.listdir(DATA_DIR)
    print(f'目录内文件数量: {len(files)}')
    parquet_files = [f for f in files if f.endswith('.parquet')]
    print(f'Parquet文件数量: {len(parquet_files)}')
    if parquet_files:
        print('前5个文件:')
        for f in parquet_files[:5]:
            print(f'  - {f}')
    else:
        print('❌ 没有找到parquet文件')
else:
    print('❌ 数据目录不存在')

print('\\n2. 测试网络连接...')
try:
    import requests
    url = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/shard_00000.parquet'
    response = requests.head(url, timeout=10)
    print(f'✅ HuggingFace连接成功: {response.status_code}')
except Exception as e:
    print(f'❌ HuggingFace连接失败: {e}')

print('\\n3. 测试本地parquet文件列表...')
try:
    parquet_paths = list_parquet_files()
    print(f'发现{len(parquet_paths)}个parquet文件')
except Exception as e:
    print(f'❌ 列表parquet文件失败: {e}')
"

echo ""
echo "=== 数据解决方案选择 ==="
echo ""
echo "选择一个解决方案："
echo "1. 下载少量数据文件进行测试"
echo "2. 创建模拟parquet数据文件"
echo "3. 使用完全模拟数据（跳过数据加载器）"
echo ""

read -p "请选择 (1/2/3): " choice

case $choice in
    1)
        echo "=== 下载测试数据 ==="
        echo "尝试下载前2个数据文件..."
        
        PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

from nanochat.dataset import download_single_file, DATA_DIR
import os

print(f'目标目录: {DATA_DIR}')
os.makedirs(DATA_DIR, exist_ok=True)

print('下载shard_00000.parquet...')
success1 = download_single_file(0)

if success1:
    print('下载shard_00001.parquet...')
    success2 = download_single_file(1)
    
    if success1 and success2:
        print('✅ 数据下载成功！现在可以运行真实数据训练')
    else:
        print('⚠️ 部分下载失败，但应该够测试用')
else:
    print('❌ 数据下载失败，建议使用模拟数据')
"
        ;;
    2)
        echo "=== 创建模拟parquet数据 ==="
        
        PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nanochat.dataset import DATA_DIR

print(f'创建模拟数据目录: {DATA_DIR}')
os.makedirs(DATA_DIR, exist_ok=True)

# 创建模拟文本数据
texts = [
    'This is a sample training text for language model training.',
    'Machine learning is a subset of artificial intelligence.',
    'Deep learning uses neural networks with multiple layers.',
    'Natural language processing helps computers understand text.',
    'Transformers are the foundation of modern language models.',
] * 200  # 重复1000次创建足够的数据

# 创建训练数据文件
train_df = pd.DataFrame({'text': texts})
train_table = pa.Table.from_pandas(train_df)

train_file = os.path.join(DATA_DIR, 'shard_00000.parquet')
pq.write_table(train_table, train_file)
print(f'✅ 创建训练文件: {train_file} ({len(texts)} 条记录)')

# 创建验证数据文件
val_texts = [
    'This is validation text for testing the model.',
    'Validation data helps evaluate model performance.',
    'Test data should be different from training data.',
] * 100

val_df = pd.DataFrame({'text': val_texts})
val_table = pa.Table.from_pandas(val_df)

val_file = os.path.join(DATA_DIR, 'shard_00001.parquet')
pq.write_table(val_table, val_file)
print(f'✅ 创建验证文件: {val_file} ({len(val_texts)} 条记录)')

print('\\n模拟数据创建完成！现在可以运行真实数据训练')
"
        ;;
    3)
        echo "=== 使用完全模拟数据 ==="
        echo "建议运行以下脚本："
        echo "./test_pure_mock_training.sh"
        echo "./quick_npu_verify.sh"
        ;;
    *)
        echo "无效选择，退出。"
        ;;
esac

echo ""
echo "=== 数据状态验证 ==="

PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
import os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

from nanochat.dataset import list_parquet_files

try:
    parquet_paths = list_parquet_files()
    print(f'✅ 当前parquet文件数量: {len(parquet_paths)}')
    for path in parquet_paths:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'  - {os.path.basename(path)}: {size_mb:.1f}MB')
    
    if len(parquet_paths) >= 2:
        print('\\n✅ 数据准备完成！可以运行以下训练脚本:')
        print('  ./debug_simple_train.sh')
        print('  ./debug_full_training.sh')
    else:
        print('\\n⚠️  数据不足，建议使用模拟数据训练脚本')
        
except Exception as e:
    print(f'❌ 检查数据失败: {e}')
"

echo ""
echo "数据下载和验证完成"