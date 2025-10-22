#!/bin/bash

echo "=== 华为服务器网络受限环境数据解决方案 ==="

# 设置基本参数
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DATA_DIR="$SCRIPT_DIR/base_data"

echo "目标目录: $BASE_DATA_DIR"
mkdir -p "$BASE_DATA_DIR"

echo "检测到网络访问受限，提供以下解决方案："
echo "1) 使用HuggingFace镜像源"
echo "2) 创建本地模拟数据（推荐）"
echo "3) 使用本地已有数据"
echo "4) 手动上传数据文件"

read -p "请选择方案 (1-4): " choice

case $choice in
    1)
        echo "=== 尝试HuggingFace镜像源 ==="
        export HF_ENDPOINT=https://hf-mirror.com
        echo "设置镜像源: $HF_ENDPOINT"
        
        echo "测试镜像连接..."
        if curl -s --connect-timeout 10 $HF_ENDPOINT > /dev/null; then
            echo "✅ 镜像源连接成功"
            
            echo "尝试通过镜像下载数据..."
            for i in {0..2}; do
                filename=$(printf "shard_%05d.parquet" $i)
                echo "下载: $filename"
                
                # 使用镜像源
                huggingface-cli download \
                    --repo-type dataset \
                    --resume-download \
                    karpathy/fineweb-edu-100b-shuffle \
                    "$filename" \
                    --local-dir "$BASE_DATA_DIR" \
                    --local-dir-use-symlinks False
                
                if [ $? -eq 0 ]; then
                    echo "✅ $filename 下载成功"
                else
                    echo "❌ $filename 下载失败"
                    break
                fi
            done
        else
            echo "❌ 镜像源连接失败，切换到方案2"
            choice=2
        fi
        ;;
    2)
        echo "=== 创建本地模拟数据 ==="
        ;;
    3)
        echo "=== 查找本地已有数据 ==="
        echo "搜索可能的数据位置..."
        find /mnt -name "*.parquet" -type f 2>/dev/null | head -10
        find /data -name "*.parquet" -type f 2>/dev/null | head -10
        find /home -name "*.parquet" -type f 2>/dev/null | head -10
        
        read -p "请输入数据文件目录路径（或按Enter跳过）: " data_path
        if [ -n "$data_path" ] && [ -d "$data_path" ]; then
            echo "复制数据到项目目录..."
            cp "$data_path"/*.parquet "$BASE_DATA_DIR/" 2>/dev/null
            echo "✅ 数据复制完成"
        else
            echo "跳过，使用方案2"
            choice=2
        fi
        ;;
    4)
        echo "=== 手动上传数据指南 ==="
        echo "1. 在本地下载FineWeb数据文件"
        echo "2. 使用scp上传到服务器:"
        echo "   scp shard_*.parquet root@server:/mnt/linxid615/bza/nanochat-npu/base_data/"
        echo "3. 或者使用文件传输工具上传"
        echo ""
        echo "按任意键继续使用模拟数据..."
        read -n 1
        choice=2
        ;;
esac

# 如果其他方案都失败，创建模拟数据
if [ "$choice" = "2" ] || [ ! -f "$BASE_DATA_DIR/shard_00000.parquet" ]; then
    echo "=== 创建本地模拟数据 ==="
    
    python3 -c "
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

base_data_dir = '$BASE_DATA_DIR'
os.makedirs(base_data_dir, exist_ok=True)

print('创建模拟FineWeb训练数据...')

# 创建更真实的训练文本
train_texts = [
    'The development of artificial intelligence has revolutionized many industries and continues to drive innovation across various sectors.',
    'Machine learning algorithms enable computers to learn patterns from data without being explicitly programmed for each specific task.',
    'Deep neural networks have achieved remarkable success in tasks such as image recognition, natural language processing, and speech synthesis.',
    'The transformer architecture has become the foundation for many state-of-the-art language models that power modern AI applications.',
    'Natural language processing techniques allow computers to understand, interpret, and generate human language in meaningful ways.',
    'Computer vision systems can analyze and interpret visual information from the world, enabling applications like autonomous vehicles.',
    'Data science combines statistical analysis, machine learning, and domain expertise to extract insights from large datasets.',
    'Cloud computing platforms provide scalable infrastructure for training and deploying machine learning models at enterprise scale.',
    'The field of robotics integrates mechanical engineering, computer science, and artificial intelligence to create autonomous systems.',
    'Ethical considerations in AI development include fairness, transparency, privacy protection, and avoiding harmful biases.',
    'Reinforcement learning enables agents to learn optimal behaviors through interaction with their environment and reward signals.',
    'Computer graphics and simulation technologies create realistic virtual environments for training AI systems and testing algorithms.',
    'Distributed computing frameworks allow researchers to process massive datasets across multiple machines efficiently.',
    'The Internet of Things connects everyday devices to networks, generating vast amounts of data for analysis and automation.',
    'Cybersecurity measures protect digital systems from threats while enabling secure data sharing and collaboration.',
    'Quantum computing promises to solve certain computational problems exponentially faster than classical computers.',
    'Bioinformatics applies computational methods to analyze biological data and understand complex biological processes.',
    'Software engineering practices ensure the reliability, maintainability, and scalability of complex AI systems.',
    'Human-computer interaction research focuses on designing intuitive interfaces between people and intelligent systems.',
    'Open source software development fosters collaboration and accelerates innovation in the global technology community.',
] * 100  # 2000条记录

# 创建训练数据文件
for shard_idx in range(5):  # 创建5个分片
    start_idx = shard_idx * 400
    end_idx = min(start_idx + 400, len(train_texts))
    shard_texts = train_texts[start_idx:end_idx]
    
    df = pd.DataFrame({'text': shard_texts})
    table = pa.Table.from_pandas(df)
    
    filename = f'shard_{shard_idx:05d}.parquet'
    filepath = os.path.join(base_data_dir, filename)
    
    pq.write_table(table, filepath)
    print(f'✅ 创建 {filename}: {len(shard_texts)} 条记录')

print(f'\\n✅ 模拟数据创建完成！')
print(f'数据目录: {base_data_dir}')
print(f'文件数量: 5个分片')
print(f'总记录数: 约2000条')
"
fi

# 验证数据
echo ""
echo "=== 验证数据文件 ==="
downloaded_files=$(ls "$BASE_DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "找到的parquet文件数量: $downloaded_files"

if [ $downloaded_files -gt 0 ]; then
    echo "✅ 数据准备成功！"
    echo "文件列表:"
    ls -lh "$BASE_DATA_DIR"/*.parquet
    
    # 测试数据读取
    echo ""
    echo "=== 测试数据读取 ==="
    python3 -c "
import pandas as pd
import os
import sys

base_data_dir = '$BASE_DATA_DIR'
files = sorted([f for f in os.listdir(base_data_dir) if f.endswith('.parquet')])

if files:
    test_file = os.path.join(base_data_dir, files[0])
    try:
        df = pd.read_parquet(test_file)
        print(f'✅ 数据读取成功: {len(df)} 行')
        print(f'列名: {list(df.columns)}')
        if 'text' in df.columns and len(df) > 0:
            print(f'示例文本: {df[\"text\"].iloc[0][:100]}...')
        print()
        print('现在可以运行训练脚本了!')
    except Exception as e:
        print(f'❌ 数据读取失败: {e}')
        sys.exit(1)
else:
    print('❌ 没有找到数据文件')
    sys.exit(1)
"
    
    echo ""
    echo "=== 下一步操作 ==="
    echo "1. 运行训练测试: ./test_local_data_training.sh"
    echo "2. 分析数据: python parquet_analysis.py"
    echo "3. 或运行简单训练: ./debug_simple_train.sh"
    
else
    echo "❌ 数据准备失败"
    exit 1
fi