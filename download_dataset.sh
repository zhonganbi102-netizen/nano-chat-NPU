#!/bin/bash

echo "=== 华为服务器HuggingFace数据集下载脚本 ==="

# 检查网络连接
echo "1. 检查网络连接..."
if ping -c 3 huggingface.co > /dev/null 2>&1; then
    echo "✅ HuggingFace网络连接正常"
else
    echo "❌ HuggingFace网络连接失败，可能需要配置代理"
fi

# 设置数据目录
DATA_DIR="/root/.cache/nanochat/base_data"
mkdir -p "$DATA_DIR"
echo "数据目录: $DATA_DIR"

# 方法1: 使用华为云镜像源（如果可用）
echo -e "\n=== 方法1: 尝试华为云镜像源 ==="
echo "设置华为云镜像..."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface

# 方法2: 使用wget直接下载
echo -e "\n=== 方法2: 直接下载parquet文件 ==="
echo "开始下载FineWeb-Edu数据集的前5个文件进行测试..."

BASE_URL="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"

download_file() {
    local index=$1
    local filename=$(printf "shard_%05d.parquet" $index)
    local filepath="$DATA_DIR/$filename"
    local url="$BASE_URL/$filename"
    
    if [ -f "$filepath" ]; then
        echo "✅ 跳过 $filename (已存在)"
        return 0
    fi
    
    echo "下载 $filename..."
    
    # 尝试多种下载方式
    for attempt in {1..3}; do
        echo "  尝试 $attempt/3..."
        
        # 方式1: wget
        if command -v wget >/dev/null 2>&1; then
            if wget -q --timeout=30 --tries=2 -O "$filepath.tmp" "$url"; then
                mv "$filepath.tmp" "$filepath"
                echo "  ✅ wget下载成功: $filename"
                return 0
            fi
        fi
        
        # 方式2: curl
        if command -v curl >/dev/null 2>&1; then
            if curl -f -L --connect-timeout 30 --max-time 300 -o "$filepath.tmp" "$url"; then
                mv "$filepath.tmp" "$filepath"
                echo "  ✅ curl下载成功: $filename"
                return 0
            fi
        fi
        
        # 清理临时文件
        rm -f "$filepath.tmp"
        
        if [ $attempt -lt 3 ]; then
            sleep_time=$((2 ** attempt))
            echo "  等待 ${sleep_time}s 后重试..."
            sleep $sleep_time
        fi
    done
    
    echo "  ❌ 下载失败: $filename"
    return 1
}

# 下载前5个文件进行测试
echo "下载测试文件..."
success_count=0
for i in {0..4}; do
    if download_file $i; then
        ((success_count++))
    fi
done

echo -e "\n=== 下载结果 ==="
echo "成功下载: $success_count/5 个文件"

# 检查下载的文件
echo -e "\n=== 检查下载的文件 ==="
ls -lh "$DATA_DIR"/*.parquet 2>/dev/null || echo "没有找到parquet文件"

# 方法3: 使用Python huggingface_hub库
echo -e "\n=== 方法3: 使用Python下载 ==="
python3 -c "
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    print('✅ huggingface_hub可用')
    
    # 设置镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    data_dir = '/root/.cache/nanochat/base_data'
    os.makedirs(data_dir, exist_ok=True)
    
    print('尝试下载单个文件进行测试...')
    try:
        file_path = hf_hub_download(
            repo_id='karpathy/fineweb-edu-100b-shuffle',
            filename='shard_00000.parquet',
            local_dir=data_dir,
            timeout=60
        )
        print(f'✅ 成功下载到: {file_path}')
    except Exception as e:
        print(f'❌ Python下载失败: {e}')
        
except ImportError:
    print('❌ huggingface_hub未安装')
    print('安装命令: pip install huggingface_hub')
"

# 方法4: 创建模拟数据作为备选
echo -e "\n=== 方法4: 创建模拟数据（备选方案） ==="
python3 -c "
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

data_dir = '/root/.cache/nanochat/base_data'
os.makedirs(data_dir, exist_ok=True)

# 检查是否已有足够数据
existing_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
if len(existing_files) >= 2:
    print(f'✅ 已有{len(existing_files)}个数据文件，无需创建模拟数据')
else:
    print('创建模拟训练数据...')
    
    # 更丰富的模拟文本
    sample_texts = [
        'The field of artificial intelligence has revolutionized modern computing and data processing.',
        'Machine learning algorithms enable computers to learn patterns from large datasets.',
        'Deep neural networks consist of multiple layers that process information hierarchically.',
        'Natural language processing allows machines to understand and generate human language.',
        'Computer vision systems can analyze and interpret visual information from images.',
        'Reinforcement learning teaches agents to make decisions through trial and error.',
        'Data science combines statistics, programming, and domain expertise to extract insights.',
        'Large language models have transformed how we approach text generation and understanding.',
        'Python is widely used in the AI and machine learning community due to its simplicity.',
        'Distributed computing enables training of large-scale machine learning models.',
        'Cloud computing provides scalable infrastructure for AI and ML workloads.',
        'GPU acceleration significantly speeds up deep learning model training.',
        'Transfer learning allows models to leverage knowledge from pre-trained networks.',
        'Attention mechanisms help models focus on relevant parts of input sequences.',
        'Transformer architectures have become the foundation for many modern AI systems.',
    ]
    
    # 创建训练数据 - 扩展到更多样本
    train_texts = []
    for base_text in sample_texts:
        # 为每个基础文本创建变体
        train_texts.append(base_text)
        train_texts.append(base_text.replace('.', ' and continues to advance rapidly.'))
        train_texts.append(f'In recent years, {base_text.lower()}')
        train_texts.append(f'{base_text} This is an important development in technology.')
    
    # 重复以达到足够的数据量
    train_texts = train_texts * 20  # 约1200条记录
    
    train_df = pd.DataFrame({'text': train_texts})
    train_table = pa.Table.from_pandas(train_df)
    train_file = os.path.join(data_dir, 'shard_00000.parquet')
    pq.write_table(train_table, train_file)
    print(f'✅ 创建训练文件: {len(train_texts)} 条记录')
    
    # 创建验证数据
    val_texts = [
        'This is validation text for evaluating model performance.',
        'Testing data helps measure model accuracy and generalization.',
        'Validation sets should be kept separate from training data.',
        'Model evaluation requires careful analysis of test results.',
        'Performance metrics guide model improvement and optimization.',
    ] * 30  # 150条记录
    
    val_df = pd.DataFrame({'text': val_texts})
    val_table = pa.Table.from_pandas(val_df)
    val_file = os.path.join(data_dir, 'shard_00001.parquet')
    pq.write_table(val_table, val_file)
    print(f'✅ 创建验证文件: {len(val_texts)} 条记录')
    
    print('✅ 模拟数据创建完成')
"

# 最终检查
echo -e "\n=== 最终检查 ==="
if [ -d "$DATA_DIR" ]; then
    file_count=$(ls -1 "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        echo "✅ 数据准备完成！"
        echo "数据目录: $DATA_DIR"
        echo "文件数量: $file_count"
        echo "文件列表:"
        ls -lh "$DATA_DIR"/*.parquet
    else
        echo "❌ 没有找到任何parquet文件"
    fi
else
    echo "❌ 数据目录不存在"
fi

echo -e "\n=== 使用建议 ==="
echo "1. 如果真实数据下载成功，可以直接使用"
echo "2. 如果下载失败，已创建模拟数据可供训练测试"
echo "3. 运行训练脚本: ./debug_simple_train.sh"
echo "4. 如需下载更多数据，可以修改循环范围 {0..4} -> {0..N}"