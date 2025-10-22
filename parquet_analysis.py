#!/usr/bin/env python3
"""
Parquet文件分析工具
用于查看和分析下载的FineWeb数据集
"""

import pandas as pd
import pyarrow.parquet as pq
import os
import sys
import argparse
from pathlib import Path

def analyze_parquet_file(file_path):
    """分析单个parquet文件"""
    print(f"\n=== 分析文件: {file_path} ===")
    
    try:
        # 使用pandas读取
        df = pd.read_parquet(file_path)
        
        print(f"✅ 文件读取成功")
        print(f"行数: {len(df):,}")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {list(df.columns)}")
        print(f"文件大小: {os.path.getsize(file_path) / (1024**2):.1f} MB")
        
        print("\n数据类型信息:")
        print(df.info())
        
        # 如果有text列，显示示例
        if 'text' in df.columns:
            print(f"\n文本列统计:")
            text_lengths = df['text'].str.len()
            print(f"平均文本长度: {text_lengths.mean():.0f} 字符")
            print(f"最短文本: {text_lengths.min()} 字符")
            print(f"最长文本: {text_lengths.max()} 字符")
            
            print(f"\n示例文本 (前3条):")
            for i, text in enumerate(df['text'].head(3)):
                print(f"[{i+1}] {text[:200]}...")
        
        # 使用pyarrow获取更多信息
        pf = pq.ParquetFile(file_path)
        print(f"\nParquet文件信息:")
        print(f"行组数: {pf.num_row_groups}")
        print(f"总行数: {pf.metadata.num_rows}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文件分析失败: {e}")
        return False

def analyze_dataset_directory(data_dir):
    """分析整个数据集目录"""
    print(f"=== 分析数据集目录: {data_dir} ===")
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 找到所有parquet文件
    parquet_files = sorted([
        f for f in os.listdir(data_dir) 
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    
    if not parquet_files:
        print("❌ 目录中没有找到parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    total_size = 0
    total_rows = 0
    
    # 分析前几个文件
    for i, filename in enumerate(parquet_files[:5]):  # 只分析前5个
        filepath = os.path.join(data_dir, filename)
        file_size = os.path.getsize(filepath)
        total_size += file_size
        
        try:
            pf = pq.ParquetFile(filepath)
            rows = pf.metadata.num_rows
            total_rows += rows
            print(f"[{i+1}] {filename}: {rows:,} 行, {file_size/(1024**2):.1f} MB")
        except Exception as e:
            print(f"[{i+1}] {filename}: 读取失败 - {e}")
    
    if len(parquet_files) > 5:
        print(f"... (还有 {len(parquet_files)-5} 个文件)")
        
        # 估算总数据量
        avg_size = total_size / min(5, len(parquet_files))
        avg_rows = total_rows / min(5, len(parquet_files))
        estimated_total_size = avg_size * len(parquet_files)
        estimated_total_rows = avg_rows * len(parquet_files)
        
        print(f"\n估算统计:")
        print(f"总文件数: {len(parquet_files)}")
        print(f"估算总大小: {estimated_total_size/(1024**3):.1f} GB")
        print(f"估算总行数: {estimated_total_rows:,.0f}")
    
    print(f"\n实际统计 (已分析文件):")
    print(f"已分析大小: {total_size/(1024**2):.1f} MB")
    print(f"已分析行数: {total_rows:,}")

def main():
    parser = argparse.ArgumentParser(description="分析FineWeb parquet数据文件")
    parser.add_argument("path", nargs="?", help="parquet文件路径或数据目录路径")
    parser.add_argument("--dir", help="指定数据目录路径")
    parser.add_argument("--file", help="指定单个文件路径") 
    
    args = parser.parse_args()
    
    # 确定要分析的路径
    target_path = None
    if args.path:
        target_path = args.path
    elif args.dir:
        target_path = args.dir
    elif args.file:
        target_path = args.file
    else:
        # 默认查看当前目录的base_data
        script_dir = Path(__file__).parent
        default_data_dir = script_dir / "base_data"
        if default_data_dir.exists():
            target_path = str(default_data_dir)
        else:
            print("请指定要分析的文件或目录路径")
            print("用法: python parquet_analysis.py <path>")
            print("或者: python parquet_analysis.py --dir <directory>")
            print("或者: python parquet_analysis.py --file <file>")
            return
    
    target_path = os.path.abspath(target_path)
    
    if os.path.isfile(target_path) and target_path.endswith('.parquet'):
        # 分析单个文件
        analyze_parquet_file(target_path)
    elif os.path.isdir(target_path):
        # 分析目录
        analyze_dataset_directory(target_path)
    else:
        print(f"❌ 无效路径: {target_path}")
        print("请提供有效的parquet文件路径或包含parquet文件的目录路径")

if __name__ == "__main__":
    main()