#!/bin/bash

echo "=== 🧹 数据清理脚本：从67GB减少到20GB ==="

BASE_DATA_DIR="./base_data"
TARGET_SIZE_GB=20
CURRENT_FILES=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
TARGET_FILES=$((20 * 1024 / 90))  # 约222个文件

echo "当前文件数: $CURRENT_FILES"
echo "目标文件数: $TARGET_FILES"
echo "需要删除: $((CURRENT_FILES - TARGET_FILES)) 个文件"

current_size=$(du -sh "$BASE_DATA_DIR" 2>/dev/null | cut -f1)
echo "当前大小: $current_size"
echo "目标大小: ${TARGET_SIZE_GB}GB"

echo ""
echo "清理策略选择:"
echo "1) 保留前面的文件 (shard_00000 到 shard_0$(printf "%04d" $((TARGET_FILES-1))))"
echo "2) 保留分散的文件 (更好的数据多样性)"
echo "3) 随机保留文件"
echo "4) 手动选择要保留的范围"
echo "5) 取消清理"

read -p "请选择清理策略 (1-5): " strategy

case $strategy in
    1)
        echo "=== 保留前${TARGET_FILES}个文件 ==="
        echo "保留: shard_00000.parquet 到 shard_$(printf "%05d" $((TARGET_FILES-1))).parquet"
        
        read -p "确认删除其他文件? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            for i in $(seq $TARGET_FILES $((CURRENT_FILES-1))); do
                filename=$(printf "shard_%05d.parquet" $i)
                if [ -f "$BASE_DATA_DIR/$filename" ]; then
                    echo "删除: $filename"
                    rm "$BASE_DATA_DIR/$filename"
                fi
            done
        fi
        ;;
        
    2)
        echo "=== 保留分散的文件 (数据多样性最佳) ==="
        step=$((CURRENT_FILES / TARGET_FILES))
        echo "保留间隔: 每${step}个文件保留1个"
        
        # 创建要保留的文件列表
        keep_files=()
        for i in $(seq 0 $((TARGET_FILES-1))); do
            file_index=$((i * step))
            filename=$(printf "shard_%05d.parquet" $file_index)
            keep_files+=("$filename")
        done
        
        echo "将保留 ${#keep_files[@]} 个文件"
        printf '%s\n' "${keep_files[@]}" | head -10
        if [ ${#keep_files[@]} -gt 10 ]; then
            echo "... (还有 $((${#keep_files[@]} - 10)) 个文件)"
        fi
        
        read -p "确认删除其他文件? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            for file in "$BASE_DATA_DIR"/*.parquet; do
                filename=$(basename "$file")
                if [[ ! " ${keep_files[@]} " =~ " ${filename} " ]]; then
                    echo "删除: $filename"
                    rm "$file"
                fi
            done
        fi
        ;;
        
    3)
        echo "=== 随机保留文件 ==="
        
        # 获取所有文件并随机排序，保留前N个
        all_files=($(ls "$BASE_DATA_DIR"/*.parquet | shuf))
        files_to_keep=("${all_files[@]:0:$TARGET_FILES}")
        
        echo "将随机保留 ${#files_to_keep[@]} 个文件"
        
        read -p "确认删除其他文件? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            for file in "$BASE_DATA_DIR"/*.parquet; do
                if [[ ! " ${files_to_keep[@]} " =~ " ${file} " ]]; then
                    filename=$(basename "$file")
                    echo "删除: $filename"
                    rm "$file"
                fi
            done
        fi
        ;;
        
    4)
        echo "=== 手动选择范围 ==="
        echo "当前文件范围: shard_00000.parquet 到 shard_$(printf "%05d" $((CURRENT_FILES-1))).parquet"
        
        read -p "起始文件编号 (0-$((CURRENT_FILES-1))): " start_num
        read -p "结束文件编号 ($start_num-$((CURRENT_FILES-1))): " end_num
        
        keep_count=$((end_num - start_num + 1))
        echo "将保留 $keep_count 个文件 (shard_$(printf "%05d" $start_num) 到 shard_$(printf "%05d" $end_num))"
        
        read -p "确认删除其他文件? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            # 删除范围外的文件
            for file in "$BASE_DATA_DIR"/*.parquet; do
                filename=$(basename "$file")
                file_num=$(echo "$filename" | sed 's/shard_0*\([0-9]*\)\.parquet/\1/')
                if [ "$file_num" -lt "$start_num" ] || [ "$file_num" -gt "$end_num" ]; then
                    echo "删除: $filename"
                    rm "$file"
                fi
            done
        fi
        ;;
        
    5)
        echo "取消清理"
        exit 0
        ;;
        
    *)
        echo "无效选择"
        exit 1
        ;;
esac

# 清理后统计
echo ""
echo "=== 📊 清理完成统计 ==="
final_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
final_size=$(du -sh "$BASE_DATA_DIR" 2>/dev/null | cut -f1)

echo "最终文件数: $final_files"
echo "最终大小: $final_size"
echo "节省空间: $((CURRENT_FILES - final_files)) 个文件"

if [ $final_files -le $((TARGET_FILES + 10)) ]; then
    echo "✅ 清理成功！数据量已优化到合理范围"
else
    echo "⚠️  文件数仍然较多，可以重新运行脚本进一步清理"
fi

# 验证剩余数据
echo ""
echo "=== 🔍 验证剩余数据 ==="
python3 -c "
import pandas as pd
import os

base_data_dir = '$BASE_DATA_DIR'
files = sorted([f for f in os.listdir(base_data_dir) if f.endswith('.parquet')])

print(f'剩余文件: {len(files)} 个')
print(f'文件范围: {files[0]} 到 {files[-1]}')

# 快速验证几个文件
total_rows = 0
for i, filename in enumerate(files[:3]):
    filepath = os.path.join(base_data_dir, filename)
    try:
        df = pd.read_parquet(filepath)
        rows = len(df)
        total_rows += rows
        if i == 0:
            print(f'示例数据正常: {df[\"text\"].iloc[0][:100]}...')
    except Exception as e:
        print(f'❌ 验证失败: {e}')

print(f'✅ 数据验证通过，可以开始训练！')
"

echo ""
echo "=== 🚀 现在可以开始训练了！==="
echo "python scripts/base_train.py --depth=12 --device_batch_size=16"