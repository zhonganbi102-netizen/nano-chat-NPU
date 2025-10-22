#!/usr/bin/env python3
"""
快速数据准备脚本 - 专为华为服务器优化
"""

import os
import sys
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time

def setup_environment():
    """设置华为服务器环境"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HOME'] = '/root/.cache/huggingface'
    print("✅ 环境配置完成")

def ensure_data_directory():
    """确保数据目录存在"""
    data_dir = Path("/root/.cache/nanochat/base_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def check_existing_data(data_dir):
    """检查已有数据"""
    parquet_files = list(data_dir.glob("*.parquet"))
    if len(parquet_files) >= 2:
        print(f"✅ 已有{len(parquet_files)}个数据文件，跳过下载")
        for f in sorted(parquet_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f}MB")
        return True
    return False

def download_single_file(url, filepath, timeout=120):
    """下载单个文件"""
    try:
        print(f"下载: {filepath.name}")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        temp_path = f"{filepath}.tmp"
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        os.rename(temp_path, filepath)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✅ 下载成功: {filepath.name} ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        # 清理临时文件
        for tmp in [f"{filepath}.tmp", filepath]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except:
                    pass
        return False

def try_download_real_data(data_dir):
    """尝试下载真实数据"""
    base_url = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    
    print("尝试下载FineWeb-Edu数据集...")
    
    success_count = 0
    for i in range(2):  # 只下载前2个文件
        filename = f"shard_{i:05d}.parquet"
        filepath = data_dir / filename
        
        if filepath.exists():
            print(f"✅ 跳过 {filename} (已存在)")
            success_count += 1
            continue
        
        url = f"{base_url}/{filename}"
        if download_single_file(url, filepath):
            success_count += 1
        else:
            break  # 如果一个失败，可能网络有问题
    
    return success_count >= 1

def create_mock_data(data_dir):
    """创建高质量模拟数据"""
    print("创建模拟训练数据...")
    
    # 更真实的训练文本
    training_texts = [
        "The development of artificial intelligence has accelerated rapidly in recent years, with breakthrough achievements in machine learning, natural language processing, and computer vision.",
        "Large language models like GPT and BERT have revolutionized how we approach text understanding and generation tasks in the field of natural language processing.",
        "Deep learning neural networks consist of multiple layers that can automatically learn hierarchical representations from raw data without manual feature engineering.",
        "Transformer architectures introduced the attention mechanism, which allows models to focus on relevant parts of input sequences for better performance.",
        "Machine learning algorithms can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning paradigms.",
        "Data preprocessing is a crucial step in machine learning pipelines, involving cleaning, normalization, and feature extraction from raw datasets.",
        "Cloud computing platforms provide scalable infrastructure for training large neural networks and processing massive datasets efficiently.",
        "Computer vision systems use convolutional neural networks to analyze and understand visual information from images and videos.",
        "Reinforcement learning agents learn optimal strategies through trial and error interactions with their environment to maximize cumulative rewards.",
        "Transfer learning enables models to leverage knowledge gained from pre-training on large datasets to perform well on specific downstream tasks.",
        "The attention mechanism in neural networks helps models identify and focus on the most relevant parts of input data during processing.",
        "Gradient descent optimization algorithms iteratively adjust model parameters to minimize loss functions and improve prediction accuracy.",
        "Batch normalization techniques help stabilize and accelerate the training process of deep neural networks by normalizing layer inputs.",
        "Dropout regularization prevents overfitting by randomly setting some neurons to zero during training, improving model generalization.",
        "Ensemble methods combine predictions from multiple models to achieve better performance than individual models alone.",
    ]
    
    # 为每个基础文本创建变体
    expanded_texts = []
    for text in training_texts:
        expanded_texts.append(text)
        expanded_texts.append(f"Recent research shows that {text.lower()}")
        expanded_texts.append(f"{text} This represents a significant advancement in the field.")
        expanded_texts.append(f"Studies have demonstrated that {text.lower()}")
        expanded_texts.append(f"In practice, {text.lower()}")
    
    # 重复以获得足够的训练数据
    final_train_texts = expanded_texts * 40  # 约3000条记录
    
    # 创建训练数据文件
    train_df = pd.DataFrame({'text': final_train_texts})
    train_table = pa.Table.from_pandas(train_df)
    train_file = data_dir / "shard_00000.parquet"
    pq.write_table(train_table, str(train_file))
    print(f"✅ 训练文件: {len(final_train_texts)} 条记录")
    
    # 创建验证数据
    validation_texts = [
        "This validation text tests the model's ability to understand and generate coherent responses.",
        "Model evaluation requires careful assessment of performance on unseen data to measure generalization.",
        "Testing datasets should be representative of real-world scenarios to ensure reliable performance metrics.",
        "Cross-validation techniques help estimate model performance and detect overfitting issues.",
        "Performance benchmarks provide standardized ways to compare different models and approaches.",
    ] * 60  # 300条记录
    
    val_df = pd.DataFrame({'text': validation_texts})
    val_table = pa.Table.from_pandas(val_df)
    val_file = data_dir / "shard_00001.parquet"
    pq.write_table(val_table, str(val_file))
    print(f"✅ 验证文件: {len(validation_texts)} 条记录")
    
    return True

def main():
    print("=== 快速数据准备脚本 ===")
    
    # 设置环境
    setup_environment()
    
    # 确保数据目录
    data_dir = ensure_data_directory()
    print(f"数据目录: {data_dir}")
    
    # 检查已有数据
    if check_existing_data(data_dir):
        return 0
    
    # 尝试下载真实数据
    print("\n1. 尝试下载真实数据...")
    if try_download_real_data(data_dir):
        print("✅ 真实数据下载成功")
    else:
        print("❌ 真实数据下载失败")
        print("\n2. 创建高质量模拟数据...")
        if not create_mock_data(data_dir):
            print("❌ 模拟数据创建失败")
            return 1
    
    # 最终检查
    print("\n=== 数据准备完成 ===")
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"数据文件数量: {len(parquet_files)}")
    
    total_size = 0
    for f in sorted(parquet_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f}MB")
    
    print(f"总大小: {total_size:.1f}MB")
    print(f"数据目录: {data_dir}")
    print("\n✅ 可以开始训练了！运行: ./debug_simple_train.sh")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())