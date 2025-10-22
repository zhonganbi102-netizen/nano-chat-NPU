#!/bin/bash

echo "=== 使用 HuggingFace CLI 下载 FineWeb 数据集 ==="

# 检查和安装依赖
echo "1. 检查 huggingface_hub 安装..."
pip show huggingface_hub > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "安装 huggingface_hub..."
    pip install -U huggingface_hub
else
    echo "✅ huggingface_hub 已安装"
fi

# 设置目标目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DATA_DIR="$SCRIPT_DIR/base_data"
echo "目标目录: $BASE_DATA_DIR"

# 创建目录
mkdir -p "$BASE_DATA_DIR"

echo "2. 开始下载 FineWeb-Edu 数据集..."

# FineWeb-Edu 数据集信息
DATASET_REPO="karpathy/fineweb-edu-100b-shuffle"
echo "数据集: $DATASET_REPO"

# 下载选项：可以选择下载部分文件
echo "选择下载选项:"
echo "1) 下载前10个分片 (约10GB, 测试用)"
echo "2) 下载前100个分片 (约100GB, 小规模训练)"
echo "3) 下载全部数据 (约1TB, 完整训练)"
echo "4) 自定义分片数量"

read -p "请选择 (1-4): " choice

case $choice in
    1)
        MAX_SHARDS=10
        echo "下载前 $MAX_SHARDS 个分片..."
        ;;
    2)
        MAX_SHARDS=100
        echo "下载前 $MAX_SHARDS 个分片..."
        ;;
    3)
        MAX_SHARDS=1823  # 全部分片 (0-1822)
        echo "下载全部 $MAX_SHARDS 个分片..."
        ;;
    4)
        read -p "请输入要下载的分片数量: " MAX_SHARDS
        echo "下载前 $MAX_SHARDS 个分片..."
        ;;
    *)
        echo "无效选择，默认下载前10个分片"
        MAX_SHARDS=10
        ;;
esac

# 使用 HuggingFace CLI 下载数据集
echo "3. 开始下载..."

# 方法1: 直接下载整个数据集到本地目录
echo "使用 huggingface-cli 下载..."

# 如果需要token，可以在这里设置
# export HUGGINGFACE_HUB_TOKEN="your_token_here"

# 下载命令
if [ "$MAX_SHARDS" -eq 1823 ]; then
    # 下载全部数据
    echo "下载全部数据集..."
    huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        "$DATASET_REPO" \
        --local-dir "$BASE_DATA_DIR" \
        --local-dir-use-symlinks False
else
    # 下载指定数量的分片
    echo "下载前 $MAX_SHARDS 个分片..."
    
    # 生成文件列表
    for i in $(seq 0 $((MAX_SHARDS-1))); do
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
            echo "❌ $filename 下载失败"
        fi
    done
fi

echo "4. 验证下载结果..."

# 检查下载的文件
downloaded_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "已下载的 parquet 文件数量: $downloaded_files"

if [ $downloaded_files -gt 0 ]; then
    echo "✅ 数据集下载成功！"
    echo "文件列表:"
    ls -lh "$BASE_DATA_DIR"/*.parquet | head -10
    
    if [ $downloaded_files -gt 10 ]; then
        echo "... (显示前10个文件)"
    fi
    
    # 测试读取第一个文件
    echo "5. 测试数据读取..."
    python3 -c "
import sys
sys.path.append('$SCRIPT_DIR')
import pandas as pd
import os

base_data_dir = '$BASE_DATA_DIR'
parquet_files = sorted([f for f in os.listdir(base_data_dir) if f.endswith('.parquet')])

if parquet_files:
    first_file = os.path.join(base_data_dir, parquet_files[0])
    print(f'测试读取: {first_file}')
    
    try:
        df = pd.read_parquet(first_file)
        print(f'✅ 文件读取成功')
        print(f'行数: {len(df)}')
        print(f'列名: {list(df.columns)}')
        if 'text' in df.columns:
            print(f'示例文本: {df[\"text\"].iloc[0][:100]}...')
    except Exception as e:
        print(f'❌ 文件读取失败: {e}')
else:
    print('❌ 没有找到 parquet 文件')
"
    
else
    echo "❌ 没有下载到任何文件，请检查网络连接或数据集访问权限"
fi

echo "6. 配置完成指南："
echo "   - 数据目录: $BASE_DATA_DIR"
echo "   - 训练脚本会自动使用这个目录中的数据"
echo "   - 如果需要更多数据，重新运行此脚本选择更大的分片数量"

echo "=== FineWeb 数据集下载完成 ==="