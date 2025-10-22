#!/bin/bash

echo "=== 🎯 合理数据量下载脚本 (20GB版本) ==="

# 设置参数
export HF_ENDPOINT=https://hf-mirror.com
BASE_DATA_DIR="./base_data"
TARGET_SIZE_GB=20
APPROX_FILE_SIZE_MB=90  # 每个文件约90MB
TARGET_FILES=$((TARGET_SIZE_GB * 1024 / APPROX_FILE_SIZE_MB))  # 约222个文件

echo "目标大小: ${TARGET_SIZE_GB}GB"
echo "预计需要文件数: $TARGET_FILES"
echo "这足够训练一个高质量的小到中型模型！"

mkdir -p "$BASE_DATA_DIR"

# 检查已下载文件
existing_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "已下载文件: $existing_files"

if [ $existing_files -ge $TARGET_FILES ]; then
    echo "✅ 已有足够数据文件，无需下载更多"
    echo "当前数据量: $(du -sh $BASE_DATA_DIR | cut -f1)"
    exit 0
fi

echo ""
echo "开始下载 $TARGET_FILES 个文件 (约${TARGET_SIZE_GB}GB)..."

# 下载策略选择
echo "下载模式:"
echo "1) 连续下载 (0-$TARGET_FILES)"
echo "2) 分散下载 (更好的数据多样性)"
echo "3) 快速下载 (并行)"

read -p "选择模式 (1-3): " mode

case $mode in
    1)
        echo "=== 连续下载模式 ==="
        for i in $(seq $existing_files $((TARGET_FILES-1))); do
            filename=$(printf "shard_%05d.parquet" $i)
            echo "下载 $filename ($((i+1))/$TARGET_FILES)..."
            hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir "$BASE_DATA_DIR"
            
            # 每下载10个文件显示进度
            if [ $((i % 10)) -eq 0 ]; then
                current_size=$(du -sh "$BASE_DATA_DIR" 2>/dev/null | cut -f1)
                echo "📊 进度: $((i+1))/$TARGET_FILES, 当前大小: $current_size"
            fi
        done
        ;;
    2)
        echo "=== 分散下载模式 ==="
        # 分散在整个数据集中选择文件，获得更好的多样性
        step=$((1823 / TARGET_FILES))
        for i in $(seq 0 $((TARGET_FILES-1))); do
            file_index=$((i * step))
            filename=$(printf "shard_%05d.parquet" $file_index)
            
            if [ ! -f "$BASE_DATA_DIR/$filename" ]; then
                echo "下载 $filename ($((i+1))/$TARGET_FILES)..."
                hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir "$BASE_DATA_DIR"
            fi
        done
        ;;
    3)
        echo "=== 快速并行下载模式 ==="
        seq $existing_files $((TARGET_FILES-1)) | xargs -n 1 -P 3 -I {} bash -c '
            filename=$(printf "shard_%05d.parquet" {})
            if [ ! -f "'$BASE_DATA_DIR'/$filename" ]; then
                echo "下载 $filename..."
                hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir "'$BASE_DATA_DIR'"
            fi
        '
        ;;
esac

# 最终统计
echo ""
echo "=== 📊 下载完成统计 ==="
final_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
total_size=$(du -sh "$BASE_DATA_DIR" 2>/dev/null | cut -f1)

echo "文件数量: $final_files"
echo "总大小: $total_size"

# 验证数据
python3 -c "
import pandas as pd
import os

base_data_dir = '$BASE_DATA_DIR'
files = sorted([f for f in os.listdir(base_data_dir) if f.endswith('.parquet')])

total_rows = 0
for i, filename in enumerate(files[:5]):  # 检查前5个文件
    filepath = os.path.join(base_data_dir, filename)
    try:
        df = pd.read_parquet(filepath)
        rows = len(df)
        total_rows += rows
        if i == 0:
            print(f'示例文本: {df[\"text\"].iloc[0][:100]}...')
    except Exception as e:
        print(f'❌ {filename} 验证失败: {e}')

estimated_total_rows = total_rows * len(files) // min(5, len(files))
print(f'\\n✅ 估计总行数: {estimated_total_rows:,}')
print(f'估计总tokens: {estimated_total_rows * 512:,} (假设平均512 tokens/行)')
print('\\n🎉 数据准备完成，可以开始训练了！')
"

echo ""
echo "=== 🚀 推荐训练命令 ==="
echo "# 单NPU基础训练 (小模型)"
echo "python scripts/base_train.py --depth=8 --device_batch_size=16"
echo ""
echo "# 多NPU训练 (中型模型)"  
echo "torchrun --standalone --nproc_per_node=8 scripts/base_train.py \\"
echo "    --depth=12 --device_batch_size=16 --total_batch_size=262144"
echo ""
echo "这个数据量足够训练出高质量的模型，而且训练时间合理！"