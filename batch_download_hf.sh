#!/bin/bash

echo "=== 华为服务器 FineWeb 数据集批量下载 ==="

# 设置镜像源（你已经设置了，但这里确保一下）
export HF_ENDPOINT=https://hf-mirror.com
echo "使用镜像源: $HF_ENDPOINT"

# 设置目标目录
BASE_DATA_DIR="./base_data"
echo "目标目录: $BASE_DATA_DIR"

# 创建目录
mkdir -p "$BASE_DATA_DIR"

echo "检查已下载的文件..."
existing_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "已有文件数: $existing_files"

if [ $existing_files -gt 0 ]; then
    echo "已下载的文件:"
    ls -lh "$BASE_DATA_DIR"/*.parquet
fi

echo ""
echo "选择下载数量:"
echo "1) 下载总共5个文件（推荐测试用）"
echo "2) 下载总共10个文件（中等规模训练）"
echo "3) 下载总共20个文件（大规模训练）"
echo "4) 自定义数量"

read -p "请选择 (1-4): " choice

case $choice in
    1) TARGET_FILES=5 ;;
    2) TARGET_FILES=10 ;;
    3) TARGET_FILES=20 ;;
    4) 
        read -p "请输入要下载的总文件数: " TARGET_FILES
        ;;
    *) 
        echo "无效选择，默认下载5个文件"
        TARGET_FILES=5
        ;;
esac

echo "目标: 总共 $TARGET_FILES 个文件"

# 下载文件
for i in $(seq 0 $((TARGET_FILES-1))); do
    filename=$(printf "shard_%05d.parquet" $i)
    filepath="$BASE_DATA_DIR/$filename"
    
    if [ -f "$filepath" ]; then
        echo "✅ $filename 已存在，跳过"
        continue
    fi
    
    echo "下载 $filename ($((i+1))/$TARGET_FILES)..."
    
    # 使用hf命令下载
    hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir "$BASE_DATA_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ $filename 下载完成"
    else
        echo "❌ $filename 下载失败"
        read -p "是否继续下载其他文件? (y/n): " continue_choice
        if [ "$continue_choice" != "y" ]; then
            break
        fi
    fi
done

# 验证结果
echo ""
echo "=== 下载完成，验证结果 ==="
final_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "最终文件数: $final_files"

if [ $final_files -gt 0 ]; then
    echo "✅ 数据下载成功！"
    echo "文件列表:"
    ls -lh "$BASE_DATA_DIR"/*.parquet
    
    echo ""
    echo "=== 数据验证 ==="
    
    # 使用Python验证数据
    python3 -c "
import pandas as pd
import os
import sys

base_data_dir = '$BASE_DATA_DIR'
files = sorted([f for f in os.listdir(base_data_dir) if f.endswith('.parquet')])

print(f'找到 {len(files)} 个parquet文件')

total_rows = 0
for i, filename in enumerate(files[:3]):  # 检查前3个文件
    filepath = os.path.join(base_data_dir, filename)
    try:
        df = pd.read_parquet(filepath)
        rows = len(df)
        total_rows += rows
        print(f'[{i+1}] {filename}: {rows:,} 行, {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
        
        if i == 0 and 'text' in df.columns:
            print(f'示例文本: {df[\"text\"].iloc[0][:200]}...')
    except Exception as e:
        print(f'❌ {filename} 读取失败: {e}')

if len(files) > 3:
    print(f'... (还有 {len(files)-3} 个文件)')

print(f'\\n前3个文件总行数: {total_rows:,}')
print('✅ 数据验证完成')
"
    
    echo ""
    echo "=== 下一步操作建议 ==="
    echo "1. 测试训练: ./test_local_data_training.sh"
    echo "2. 分析数据: python parquet_analysis.py"
    echo "3. 运行完整训练: ./debug_simple_train.sh"
    
else
    echo "❌ 没有成功下载任何文件"
    echo "请检查网络连接和镜像源设置"
fi