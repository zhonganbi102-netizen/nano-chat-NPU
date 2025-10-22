#!/bin/bash

echo "=== 🔍 数据文件路径诊断脚本 ==="

echo "1. 检查当前目录..."
pwd

echo "2. 检查base_data目录..."
if [ -d "./base_data" ]; then
    echo "✅ ./base_data 目录存在"
    file_count=$(ls ./base_data/*.parquet 2>/dev/null | wc -l)
    echo "   文件数量: $file_count"
    if [ $file_count -gt 0 ]; then
        echo "   文件列表 (前5个):"
        ls -la ./base_data/*.parquet | head -5
        total_size=$(du -sh ./base_data 2>/dev/null | cut -f1)
        echo "   总大小: $total_size"
    fi
else
    echo "❌ ./base_data 目录不存在"
fi

echo "3. 检查其他可能的数据目录..."
find . -name "*.parquet" -type f 2>/dev/null | head -10

echo "4. 检查nanochat的数据目录配置..."
python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

try:
    from nanochat.common import get_base_dir
    from nanochat.dataset import DATA_DIR
    import os
    
    base_dir = get_base_dir()
    print(f'nanochat base_dir: {base_dir}')
    print(f'nanochat DATA_DIR: {DATA_DIR}')
    
    print(f'base_dir存在: {os.path.exists(base_dir)}')
    print(f'DATA_DIR存在: {os.path.exists(DATA_DIR)}')
    
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
        print(f'DATA_DIR中的文件数: {len(files)}')
        if len(files) > 0:
            print(f'第一个文件: {files[0]}')
    
except Exception as e:
    print(f'错误: {e}')
    import traceback
    traceback.print_exc()
"

echo "5. 解决方案建议..."

# 检查当前目录的数据文件
current_files=$(ls ./base_data/*.parquet 2>/dev/null | wc -l)

if [ $current_files -gt 0 ]; then
    echo "发现当前目录有数据文件，建议创建符号链接："
    
    python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

try:
    from nanochat.common import get_base_dir
    from nanochat.dataset import DATA_DIR
    import os
    
    base_dir = get_base_dir()
    current_data = './base_data'
    
    print(f'当前数据目录: {os.path.abspath(current_data)}')
    print(f'nanochat期望目录: {DATA_DIR}')
    
    if not os.path.exists(DATA_DIR):
        print('\\n建议执行以下命令创建符号链接:')
        print(f'mkdir -p {os.path.dirname(DATA_DIR)}')
        print(f'ln -sf {os.path.abspath(current_data)} {DATA_DIR}')
    else:
        print('\\n目标目录已存在，建议复制文件:')
        print(f'cp ./base_data/*.parquet {DATA_DIR}/')
        
except Exception as e:
    print(f'配置检查失败: {e}')
"
    
else
    echo "❌ 当前目录没有数据文件，请检查下载状态"
fi

echo ""
echo "=== 🚀 快速修复命令 ==="
echo "# 如果数据在当前目录的base_data中，创建符号链接:"
echo "python3 -c \"
import sys, os
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')
from nanochat.dataset import DATA_DIR
os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
if os.path.exists('./base_data') and not os.path.exists(DATA_DIR):
    os.symlink(os.path.abspath('./base_data'), DATA_DIR)
    print(f'✅ 创建符号链接: ./base_data -> {DATA_DIR}')
elif os.path.exists(DATA_DIR):
    import shutil
    if os.path.exists('./base_data'):
        for f in os.listdir('./base_data'):
            if f.endswith('.parquet'):
                shutil.copy2(os.path.join('./base_data', f), DATA_DIR)
        print('✅ 复制数据文件到目标目录')
else:
    print('❌ 找不到数据文件')
\""