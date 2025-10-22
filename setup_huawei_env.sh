#!/bin/bash

echo "=== 华为服务器环境配置脚本 ==="

# 1. 设置HuggingFace镜像和代理
echo "1. 配置HuggingFace环境..."

# 华为云HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# 如果有代理设置
if [ -n "$HTTP_PROXY" ] || [ -n "$http_proxy" ]; then
    echo "检测到代理设置"
    echo "HTTP_PROXY: $HTTP_PROXY"
    echo "HTTPS_PROXY: $HTTPS_PROXY"
fi

# 2. 检查和安装必要的Python包
echo -e "\n2. 检查Python依赖..."

python3 -c "
import sys
required_packages = [
    'requests', 'pandas', 'pyarrow', 'huggingface_hub', 
    'datasets', 'tokenizers'
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg} (缺失)')
        missing.append(pkg)

if missing:
    print(f'\\n需要安装: {\" \".join(missing)}')
    print('安装命令:')
    for pkg in missing:
        if pkg == 'huggingface_hub':
            print(f'  pip install {pkg}[cli]')
        else:
            print(f'  pip install {pkg}')
else:
    print('\\n✅ 所有依赖已满足')
"

# 3. 测试网络连接
echo -e "\n3. 测试网络连接..."

test_urls=(
    "https://huggingface.co"
    "https://hf-mirror.com" 
    "https://pypi.org"
    "https://files.pythonhosted.org"
)

for url in "${test_urls[@]}"; do
    if curl -f -s --connect-timeout 10 --max-time 20 "$url" >/dev/null; then
        echo "✅ $url"
    else
        echo "❌ $url"
    fi
done

# 4. 创建华为服务器专用的数据下载函数
echo -e "\n4. 创建华为服务器专用下载脚本..."

cat > /tmp/huawei_download.py << 'EOF'
#!/usr/bin/env python3
"""
华为服务器专用的HuggingFace数据集下载脚本
"""

import os
import sys
import requests
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# 配置华为云镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/.cache/huggingface'

def download_with_retry(url, filepath, max_retries=3):
    """使用重试机制下载文件"""
    for attempt in range(max_retries):
        try:
            print(f"  尝试 {attempt + 1}/{max_retries}: {os.path.basename(filepath)}")
            
            # 使用requests下载
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # 写入临时文件
            temp_path = f"{filepath}.tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 移动到最终位置
            os.rename(temp_path, filepath)
            print(f"  ✅ 下载成功: {os.path.basename(filepath)}")
            return True
            
        except Exception as e:
            print(f"  ❌ 尝试 {attempt + 1} 失败: {e}")
            # 清理临时文件
            for tmp_file in [f"{filepath}.tmp", filepath]:
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except:
                        pass
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
    
    return False

def download_fineweb_sample():
    """下载FineWeb-Edu样本数据"""
    base_url = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    data_dir = Path("/root/.cache/nanochat/base_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始下载FineWeb-Edu样本数据...")
    
    # 下载前3个文件
    success_count = 0
    for i in range(3):
        filename = f"shard_{i:05d}.parquet"
        filepath = data_dir / filename
        
        if filepath.exists():
            print(f"✅ 跳过 {filename} (已存在)")
            success_count += 1
            continue
        
        url = f"{base_url}/{filename}"
        if download_with_retry(url, str(filepath)):
            success_count += 1
    
    print(f"下载完成: {success_count}/3 个文件")
    return success_count > 0

def create_mock_data():
    """创建模拟数据"""
    data_dir = Path("/root/.cache/nanochat/base_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("创建模拟数据...")
    
    # 丰富的样本文本
    base_texts = [
        "Artificial intelligence is transforming the way we work and live.",
        "Machine learning algorithms can identify patterns in complex datasets.",
        "Deep learning models require substantial computational resources for training.",
        "Natural language processing enables computers to understand human communication.",
        "Computer vision systems can analyze and interpret visual information.",
        "Reinforcement learning agents learn through interaction with their environment.",
        "Data science combines domain expertise with analytical and programming skills.",
        "Large language models demonstrate remarkable capabilities in text generation.",
        "Cloud computing provides scalable infrastructure for AI applications.",
        "Neural networks are inspired by the structure of biological brains.",
    ]
    
    # 生成更多变体
    train_texts = []
    for text in base_texts:
        train_texts.append(text)
        train_texts.append(f"Recent advances show that {text.lower()}")
        train_texts.append(f"{text} This represents a significant breakthrough.")
        train_texts.append(f"Research indicates that {text.lower()}")
    
    # 重复以获得足够的数据
    train_texts = train_texts * 50  # 约2000条记录
    
    # 创建训练文件
    train_df = pd.DataFrame({'text': train_texts})
    train_table = pa.Table.from_pandas(train_df)
    train_file = data_dir / "shard_00000.parquet"
    pq.write_table(train_table, str(train_file))
    print(f"✅ 创建训练文件: {len(train_texts)} 条记录")
    
    # 创建验证文件
    val_texts = [
        "This is validation text for model evaluation and testing.",
        "Performance assessment requires separate validation datasets.",
        "Model generalization depends on diverse and representative test data.",
    ] * 100  # 300条记录
    
    val_df = pd.DataFrame({'text': val_texts})
    val_table = pa.Table.from_pandas(val_df)
    val_file = data_dir / "shard_00001.parquet"
    pq.write_table(val_table, str(val_file))
    print(f"✅ 创建验证文件: {len(val_texts)} 条记录")
    
    return True

def main():
    print("=== 华为服务器数据准备脚本 ===")
    
    # 尝试下载真实数据
    if download_fineweb_sample():
        print("✅ 真实数据下载成功")
    else:
        print("❌ 真实数据下载失败，创建模拟数据...")
        create_mock_data()
    
    # 检查最终结果
    data_dir = Path("/root/.cache/nanochat/base_data")
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if parquet_files:
        print(f"\n✅ 数据准备完成！")
        print(f"数据目录: {data_dir}")
        print(f"文件数量: {len(parquet_files)}")
        for f in sorted(parquet_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f}MB")
    else:
        print("❌ 数据准备失败")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

echo "✅ 华为服务器下载脚本已创建: /tmp/huawei_download.py"

# 5. 配置环境变量文件
echo -e "\n5. 创建环境配置文件..."

cat > ~/.huawei_ai_env << 'EOF'
# 华为服务器AI环境配置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# PyTorch和NPU相关
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1

# 训练相关缓存
export NANOCHAT_CACHE=/root/.cache/nanochat
EOF

echo "✅ 环境配置已保存到: ~/.huawei_ai_env"
echo "使用方法: source ~/.huawei_ai_env"

echo -e "\n=== 完成配置 ==="
echo "下一步操作："
echo "1. 运行华为下载脚本: python3 /tmp/huawei_download.py"
echo "2. 或者运行综合下载脚本: ./download_dataset.sh"
echo "3. 加载环境变量: source ~/.huawei_ai_env"
echo "4. 开始训练: ./debug_simple_train.sh"