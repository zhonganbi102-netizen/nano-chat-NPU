#!/bin/bash

# 快速数据下载脚本
# Quick data download script

set -e
echo "=== 下载NanoChat训练数据 ==="

# 创建数据目录
mkdir -p ~/.cache/nanochat/base_data/
cd ~/.cache/nanochat/base_data/

echo "开始下载FineWeb数据集前5个分片..."

# 下载前5个分片用于训练测试
for i in {0..4}; do
    filename=$(printf "shard_%05d.parquet" $i)
    url="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/$filename"
    
    if [ ! -f "$filename" ]; then
        echo "下载 $filename..."
        wget "$url" -O "$filename" || curl -L "$url" -o "$filename"
        echo "✅ $filename 下载完成"
    else
        echo "⚡ $filename 已存在，跳过"
    fi
done

echo "检查下载结果:"
ls -lh *.parquet

echo "✅ 数据下载完成！"
echo "现在可以运行训练了："
echo "  cd /mnt/linxid615/bza/nanochat-npu"
echo "  ./simple_base_train.sh"