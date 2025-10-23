#!/bin/bash

# 使用hf命令行工具下载数据集
# Download dataset using hf CLI tool

set -e
echo "=== 下载NanoChat训练数据 (使用hf CLI) ==="

# 设置HuggingFace镜像站
export HF_ENDPOINT=https://hf-mirror.com
echo "使用HuggingFace镜像站: $HF_ENDPOINT"

# 检查磁盘空间
echo "=== 检查磁盘空间 ==="
df -h

# 安装hf命令行工具
echo "检查并安装huggingface_hub CLI..."
pip install -U "huggingface_hub[cli]" --quiet

# 创建数据目录
DATA_DIR=~/.cache/nanochat/base_data
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "开始下载前5个数据文件..."

# 下载前5个分片
for i in {0..4}; do
    filename=$(printf "shard_%05d.parquet" $i)
    
    if [ -f "$filename" ]; then
        echo "⚡ $filename 已存在，跳过"
        continue
    fi
    
    echo "📥 下载 $filename..."
    
    # 使用hf下载命令
    if hf download --repo-type dataset karpathy/fineweb-edu-100b-shuffle "$filename" --local-dir .; then
        echo "✅ $filename 下载完成"
    else
        echo "❌ $filename 下载失败，尝试其他方法..."
        
        # 备用方法：使用huggingface_hub Python API
        python3 -c "
from huggingface_hub import hf_hub_download
import os
try:
    file_path = hf_hub_download(
        repo_id='karpathy/fineweb-edu-100b-shuffle',
        filename='$filename',
        local_dir='.',
        cache_dir=None
    )
    print('✅ $filename 下载成功（备用方法）')
except Exception as e:
    print(f'❌ $filename 下载失败: {e}')
"
    fi
done

echo ""
echo "📊 检查下载结果:"
ls -lh *.parquet 2>/dev/null || echo "没有找到parquet文件"

# 计算总大小
if ls *.parquet 1> /dev/null 2>&1; then
    total_size=$(du -sh *.parquet | awk '{sum+=$1} END {print sum}')
    file_count=$(ls *.parquet | wc -l)
    echo "✅ 数据下载完成！"
    echo "📁 文件数量: $file_count"
    echo "💾 总大小: $(du -sh . | cut -f1)"
    echo ""
    echo "🎯 现在可以运行训练:"
    echo "  cd /mnt/linxid615/bza/nanochat-npu"
    echo "  ./simple_base_train.sh"
else
    echo "❌ 没有成功下载任何文件"
    echo "请检查网络连接和数据集存在性"
    exit 1
fi