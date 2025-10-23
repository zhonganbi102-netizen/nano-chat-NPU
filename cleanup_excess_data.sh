#!/bin/bash

# 清理多余的FineWeb数据文件
# 保留前300个连续文件，删除其他文件

set -e

echo "=== FineWeb数据清理脚本 ==="

cd /mnt/linxid615/bza/nanochat-npu/base_data

echo "清理前统计："
echo "文件总数: $(ls shard_*.parquet | wc -l)"
echo "总大小: $(du -sh . | cut -f1)"

# 保留 shard_00000 到 shard_00299 (300个文件)
# 删除 shard_00300 及以上的文件

echo ""
echo "开始清理..."
echo "保留: shard_00000.parquet 到 shard_00299.parquet"
echo "删除: shard_00300.parquet 及以上的所有文件"

# 删除 shard_00300 到 shard_00999
for i in {300..999}; do
    filename=$(printf "shard_%05d.parquet" $i)
    if [ -f "$filename" ]; then
        rm "$filename"
        echo "删除: $filename"
    fi
done

# 删除 shard_01000 及以上
for i in {1000..9999}; do
    filename=$(printf "shard_%05d.parquet" $i)
    if [ -f "$filename" ]; then
        rm "$filename"
        echo "删除: $filename"
    fi
done

echo ""
echo "清理完成！"
echo ""
echo "清理后统计："
remaining_files=$(ls shard_*.parquet 2>/dev/null | wc -l || echo "0")
echo "剩余文件数: $remaining_files"
echo "剩余大小: $(du -sh . | cut -f1)"

echo ""
if [ "$remaining_files" -ge 200 ]; then
    echo "✅ 数据清理成功！剩余$remaining_files个文件，足够进行高质量训练"
    echo "预计训练时间: 2-4小时 (相比之前的10+小时大大缩短)"
else
    echo "⚠️  剩余文件较少，但仍可进行训练"
fi

echo ""
echo "可以开始训练: ./train_with_fineweb.sh"
