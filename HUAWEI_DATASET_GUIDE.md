# 华为服务器 HuggingFace 数据集下载指南

## 方法1: 使用 HuggingFace CLI（推荐）

### 1. 安装 HuggingFace Hub
```bash
pip install -U huggingface_hub
```

### 2. 下载 FineWeb 数据集

#### 快速测试（推荐先试这个）
```bash
./quick_download_data.sh
```

#### 完整下载选项
```bash
./download_fineweb_hf.sh
```

#### 手动下载特定文件
```bash
# 下载前10个分片
for i in {0..9}; do
    filename=$(printf "shard_%05d.parquet" $i)
    huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        karpathy/fineweb-edu-100b-shuffle \
        "$filename" \
        --local-dir ./base_data \
        --local-dir-use-symlinks False
done
```

## 方法2: 使用 datasets 库

### 直接在Python中加载
```python
from datasets import load_dataset

# 下载到缓存
dataset = load_dataset("karpathy/fineweb-edu-100b-shuffle", split="train")

# 或者下载到指定目录
dataset = load_dataset(
    "karpathy/fineweb-edu-100b-shuffle", 
    split="train",
    cache_dir="./base_data"
)
```

## 方法3: 备用下载（如果HF CLI失败）

### 使用 wget 直接下载
```bash
mkdir -p base_data
cd base_data

# 下载前5个文件
for i in {0..4}; do
    filename=$(printf "shard_%05d.parquet" $i)
    wget "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/$filename"
done
```

## 数据验证和分析

### 使用分析工具
```bash
# 分析整个数据目录
python parquet_analysis.py ./base_data

# 分析单个文件
python parquet_analysis.py ./base_data/shard_00000.parquet
```

### 手动验证
```python
import pandas as pd
import os

# 检查下载的文件
data_dir = "./base_data"
files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
print(f"下载的文件数: {len(files)}")

# 读取第一个文件测试
if files:
    df = pd.read_parquet(os.path.join(data_dir, files[0]))
    print(f"行数: {len(df)}, 列: {list(df.columns)}")
    print(f"示例: {df['text'].iloc[0][:100]}")
```

## 训练数据配置

### 确保目录结构正确
```
nanochat-npu/
├── base_data/           # 数据目录
│   ├── shard_00000.parquet
│   ├── shard_00001.parquet
│   └── ...
├── nanochat/
│   ├── dataset.py      # 会自动查找 base_data 目录
│   └── dataloader.py
└── train_scripts/
```

### 测试数据加载
```bash
# 运行训练前先测试数据加载
python -c "
from nanochat.dataset import list_parquet_files, parquets_iter_batched
files = list_parquet_files()
print(f'找到 {len(files)} 个数据文件')
for batch in parquets_iter_batched('train'):
    print(f'批次大小: {len(batch)}')
    break
"
```

## 常见问题解决

### 1. 网络连接问题
```bash
# 设置代理（如果需要）
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 或者设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 磁盘空间不足
```bash
# 检查磁盘空间
df -h

# 只下载部分数据
# 修改 quick_download_data.sh 中的循环范围
for i in {0..2}; do  # 只下载前3个文件
```

### 3. 权限问题
```bash
# 确保有写权限
chmod 755 base_data
chmod 644 base_data/*.parquet
```

### 4. Token 认证（如果需要）
```bash
# 设置 HuggingFace token
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# 或者登录
huggingface-cli login
```

## 推荐工作流程

1. **快速测试**: 
   ```bash
   ./quick_download_data.sh
   ```

2. **验证数据**:
   ```bash
   python parquet_analysis.py
   ```

3. **测试训练**:
   ```bash
   ./test_local_data_training.sh
   ```

4. **如果成功，下载更多数据**:
   ```bash
   ./download_fineweb_hf.sh
   ```

## 文件说明

- `quick_download_data.sh` - 快速下载前5个分片
- `download_fineweb_hf.sh` - 完整下载脚本，支持选择数量
- `parquet_analysis.py` - 数据分析工具
- `test_local_data_training.sh` - 训练测试脚本

## 数据集信息

- **数据集**: karpathy/fineweb-edu-100b-shuffle
- **总大小**: ~1TB (1823个分片)
- **单个分片**: ~500MB
- **测试用**: 5-10个分片就足够
- **格式**: Parquet文件，包含 'text' 列