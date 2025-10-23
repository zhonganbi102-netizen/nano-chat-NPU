#!/bin/bash

# FineWeb数据集下载脚本 - NPU优化版
# 下载200个文件进行完整训练

set -e

echo "=== FineWeb数据集下载脚本 ==="

# 设置环境
export HF_ENDPOINT=https://hf-mirror.com

# 检查磁盘空间
echo "=== 检查磁盘空间 ==="
df -h
echo ""

# 获取可用空间 (GB)
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "当前可用空间: ${available_space}GB"

# 估算所需空间 (每个文件约100-200MB，200个文件约20-40GB)
required_space=50
if [ "$available_space" -lt "$required_space" ]; then
    echo "❌ 磁盘空间不足！需要至少${required_space}GB，当前只有${available_space}GB"
    echo "请清理磁盘空间后重试"
    exit 1
fi

# 创建数据目录
echo "=== 创建数据目录 ==="
mkdir -p ./base_data
cd ./base_data

# 检查已下载文件
existing_files=$(ls shard_*.parquet 2>/dev/null | wc -l || echo "0")
echo "已存在文件数量: $existing_files"

# 设置下载参数
total_files=200
start_shard=5
end_shard=$((start_shard + total_files - 1))
concurrent_downloads=3  # 降低并发数避免网络拥堵

echo "=== 开始下载FineWeb数据集 ==="
echo "目标文件数量: $total_files"
echo "文件范围: shard_$(printf "%05d" $start_shard).parquet 到 shard_$(printf "%05d" $end_shard).parquet"
echo "并发下载数: $concurrent_downloads"
echo ""

# 创建下载进度文件
progress_file="download_progress.txt"
echo "0" > $progress_file

# 并行下载函数
download_shard() {
    local shard_num=$1
    local filename=$(printf "shard_%05d.parquet" $shard_num)
    
    if [ -f "$filename" ]; then
        echo "⏭️  跳过已存在: $filename"
        return 0
    fi
    
    echo "📥 开始下载: $filename"
    
    # 使用重试机制
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir . 2>/dev/null; then
            echo "✅ 完成: $filename"
            
            # 更新进度
            local current_progress=$(cat $progress_file)
            echo $((current_progress + 1)) > $progress_file
            local total_downloaded=$(cat $progress_file)
            local percentage=$((total_downloaded * 100 / total_files))
            echo "📊 进度: $total_downloaded/$total_files ($percentage%)"
            
            return 0
        else
            retry=$((retry + 1))
            echo "⚠️  重试 $retry/$max_retries: $filename"
            sleep 2
        fi
    done
    
    echo "❌ 失败: $filename (已重试$max_retries次)"
    return 1
}

# 导出函数供xargs使用
export -f download_shard
export progress_file
export total_files

# 开始并行下载
echo "开始并行下载..."
seq $start_shard $end_shard | xargs -n 1 -P $concurrent_downloads -I {} bash -c 'download_shard {}'

# 检查下载结果
echo ""
echo "=== 下载完成检查 ==="
downloaded_count=$(ls shard_*.parquet 2>/dev/null | wc -l || echo "0")
echo "实际下载文件数: $downloaded_count"

if [ "$downloaded_count" -ge 150 ]; then
    echo "✅ 下载成功！已获得$downloaded_count个文件，足够进行训练"
elif [ "$downloaded_count" -ge 100 ]; then
    echo "⚠️  部分成功：已获得$downloaded_count个文件，可以进行训练但数据量较少"
else
    echo "❌ 下载不足：只有$downloaded_count个文件，建议重新运行或检查网络"
    exit 1
fi

# 显示文件大小统计
echo ""
echo "=== 数据集统计 ==="
total_size=$(du -sh . | cut -f1)
echo "总大小: $total_size"
echo "文件数量: $downloaded_count"
echo "平均文件大小: $(du -sm . | cut -f1 | awk -v count=$downloaded_count '{printf "%.1fMB", $1/count}')"

# 清理进度文件
rm -f $progress_file

echo ""
echo "🎉 数据集下载完成！"
echo "数据位置: $(pwd)"
echo "可以开始训练了：./train_with_fineweb.sh"
