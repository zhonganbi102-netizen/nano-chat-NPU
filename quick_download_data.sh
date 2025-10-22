#!/bin/bash

echo "=== 华为服务器 FineWeb 数据集快速下载 ==="

# 设置基本参数
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DATA_DIR="$SCRIPT_DIR/base_data"
DATASET_REPO="karpathy/fineweb-edu-100b-shuffle"

echo "目标目录: $BASE_DATA_DIR"
mkdir -p "$BASE_DATA_DIR"

# 检查和安装依赖
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装 huggingface_hub..."
    pip install -U huggingface_hub
fi

# 快速选项：只下载前几个文件用于测试
echo "快速下载前5个分片进行测试..."

for i in {0..4}; do
    filename=$(printf "shard_%05d.parquet" $i)
    echo "下载: $filename"
    
    huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        "$DATASET_REPO" \
        "$filename" \
        --local-dir "$BASE_DATA_DIR" \
        --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✅ $filename 下载完成"
    else
        echo "❌ $filename 下载失败，尝试备用方法..."
        
        # 备用方法：直接下载
        cd "$BASE_DATA_DIR"
        wget "https://huggingface.co/datasets/$DATASET_REPO/resolve/main/$filename" -O "$filename" --timeout=30
        cd - > /dev/null
    fi
done

# 验证下载结果
downloaded_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "已下载文件数: $downloaded_files"

if [ $downloaded_files -gt 0 ]; then
    echo "✅ 测试数据下载成功！"
    ls -lh "$BASE_DATA_DIR"/*.parquet
    
    # 快速验证
    python3 -c "
import pandas as pd
import os
import sys

base_data_dir = '$BASE_DATA_DIR'
files = [f for f in os.listdir(base_data_dir) if f.endswith('.parquet')]

if files:
    test_file = os.path.join(base_data_dir, files[0])
    try:
        df = pd.read_parquet(test_file)
        print(f'✅ 数据验证成功: {len(df)} 行')
        print(f'列: {list(df.columns)}')
    except Exception as e:
        print(f'❌ 数据验证失败: {e}')
        sys.exit(1)
else:
    print('❌ 没有找到数据文件')
    sys.exit(1)
"
    
    echo "现在可以运行训练脚本了！"
else
    echo "❌ 下载失败，请检查网络连接"
    exit 1
fi