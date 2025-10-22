#!/bin/bash

echo "=== 🚀 FineWeb 完整数据集下载脚本 ==="
echo "警告：这将下载约1TB的数据（1823个文件）"
echo "确保你有足够的磁盘空间和时间！"

# 设置基本参数
export HF_ENDPOINT=https://hf-mirror.com
BASE_DATA_DIR="./base_data"
TOTAL_FILES=1823  # 0-1822
BATCH_SIZE=10     # 每批下载10个文件
LOG_FILE="download.log"

echo "镜像源: $HF_ENDPOINT"
echo "目标目录: $BASE_DATA_DIR"
echo "总文件数: $TOTAL_FILES"
echo "日志文件: $LOG_FILE"

# 创建目录
mkdir -p "$BASE_DATA_DIR"

# 检查已下载文件
existing_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "已下载文件: $existing_files"

# 确认下载
echo ""
echo "下载选项:"
echo "1) 快速模式：并行下载，速度快但占用带宽大"
echo "2) 稳定模式：逐个下载，稳定但较慢" 
echo "3) 断点续传：从上次中断处继续"
echo "4) 自定义范围：指定下载文件范围"
echo "5) 退出"

read -p "请选择模式 (1-5): " mode

case $mode in
    1)
        echo "=== 快速并行下载模式 ==="
        PARALLEL_JOBS=5
        ;;
    2)
        echo "=== 稳定逐个下载模式 ==="
        PARALLEL_JOBS=1
        ;;
    3)
        echo "=== 断点续传模式 ==="
        PARALLEL_JOBS=3
        ;;
    4)
        echo "=== 自定义范围模式 ==="
        read -p "起始文件编号 (0-1822): " START_NUM
        read -p "结束文件编号 (0-1822): " END_NUM
        read -p "并行任务数 (1-10): " PARALLEL_JOBS
        TOTAL_FILES=$((END_NUM - START_NUM + 1))
        ;;
    5)
        echo "退出下载"
        exit 0
        ;;
    *)
        echo "无效选择，使用稳定模式"
        PARALLEL_JOBS=1
        ;;
esac

# 设置起始和结束点
START_NUM=${START_NUM:-0}
END_NUM=${END_NUM:-1822}

echo ""
echo "=== 下载配置 ==="
echo "起始文件: shard_$(printf "%05d" $START_NUM).parquet"
echo "结束文件: shard_$(printf "%05d" $END_NUM).parquet"
echo "文件范围: $START_NUM - $END_NUM (共 $TOTAL_FILES 个文件)"
echo "并行任务: $PARALLEL_JOBS"

read -p "确认开始下载? (y/N): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "取消下载"
    exit 0
fi

# 开始下载
echo ""
echo "=== 🚀 开始下载 $(date) ===" | tee -a "$LOG_FILE"

# 创建下载函数
download_file() {
    local file_index=$1
    local filename=$(printf "shard_%05d.parquet" $file_index)
    local filepath="$BASE_DATA_DIR/$filename"
    
    # 检查文件是否已存在
    if [ -f "$filepath" ] && [ -s "$filepath" ]; then
        echo "✅ $filename 已存在，跳过" | tee -a "$LOG_FILE"
        return 0
    fi
    
    echo "📥 开始下载 $filename..." | tee -a "$LOG_FILE"
    local start_time=$(date +%s)
    
    # 下载文件
    if hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir "$BASE_DATA_DIR" 2>>"$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local size=$(du -h "$filepath" | cut -f1)
        echo "✅ $filename 完成 (${duration}s, ${size})" | tee -a "$LOG_FILE"
        return 0
    else
        echo "❌ $filename 失败" | tee -a "$LOG_FILE"
        # 清理失败的文件
        rm -f "$filepath" 2>/dev/null
        return 1
    fi
}

# 导出函数以供并行使用
export -f download_file
export BASE_DATA_DIR LOG_FILE HF_ENDPOINT

# 并行下载
if [ "$PARALLEL_JOBS" -gt 1 ]; then
    echo "使用 $PARALLEL_JOBS 个并行任务下载..."
    seq $START_NUM $END_NUM | xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c 'download_file {}'
else
    echo "逐个下载文件..."
    for i in $(seq $START_NUM $END_NUM); do
        download_file $i
        
        # 每下载10个文件显示进度
        if [ $((i % 10)) -eq 0 ]; then
            completed=$((i - START_NUM + 1))
            percentage=$((completed * 100 / TOTAL_FILES))
            echo "📊 进度: $completed/$TOTAL_FILES ($percentage%)" | tee -a "$LOG_FILE"
        fi
    done
fi

# 最终统计
echo ""
echo "=== 📊 下载完成统计 $(date) ===" | tee -a "$LOG_FILE"

final_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
total_size=$(du -sh "$BASE_DATA_DIR" 2>/dev/null | cut -f1)

echo "最终文件数: $final_files" | tee -a "$LOG_FILE"
echo "总大小: $total_size" | tee -a "$LOG_FILE"

if [ $final_files -eq $TOTAL_FILES ]; then
    echo "🎉 恭喜！所有文件下载完成！" | tee -a "$LOG_FILE"
else
    missing=$((TOTAL_FILES - final_files))
    echo "⚠️  还有 $missing 个文件未完成" | tee -a "$LOG_FILE"
    echo "可以重新运行此脚本进行断点续传" | tee -a "$LOG_FILE"
fi

# 验证数据
echo ""
echo "=== 🔍 数据验证 ==="
python3 -c "
import pandas as pd
import os
import random

base_data_dir = '$BASE_DATA_DIR'
files = sorted([f for f in os.listdir(base_data_dir) if f.endswith('.parquet')])

print(f'验证 {len(files)} 个文件...')

# 随机检查几个文件
sample_files = random.sample(files, min(5, len(files)))
total_rows = 0

for filename in sample_files:
    filepath = os.path.join(base_data_dir, filename)
    try:
        df = pd.read_parquet(filepath)
        rows = len(df)
        total_rows += rows
        print(f'✅ {filename}: {rows:,} 行')
    except Exception as e:
        print(f'❌ {filename}: 验证失败 - {e}')

print(f'\\n抽样验证完成，平均每文件约 {total_rows // len(sample_files):,} 行')
print('🎉 数据集准备就绪，可以开始大规模训练！')
"

echo ""
echo "=== 📋 下一步建议 ==="
echo "1. 检查磁盘空间: df -h"
echo "2. 查看日志: tail -f $LOG_FILE"
echo "3. 开始训练: ./debug_simple_train.sh"
echo "4. 分析数据: python parquet_analysis.py"

echo ""
echo "🚀 FineWeb 完整数据集下载任务完成！"