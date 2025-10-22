#!/bin/bash

echo "=== 检查NanoChat数据文件 ==="

cd /mnt/linxid615/bza/nanochat-npu

echo "1. 检查基础目录结构..."
ls -la

echo -e "\n2. 检查是否有数据目录..."
if [ -d "data" ]; then
    echo "✅ data目录存在"
    ls -la data/
else
    echo "❌ data目录不存在"
fi

if [ -d "tokenized_data" ]; then
    echo "✅ tokenized_data目录存在"
    ls -la tokenized_data/
else
    echo "❌ tokenized_data目录不存在"
fi

echo -e "\n3. 检查脚本目录..."
if [ -d "scripts" ]; then
    echo "✅ scripts目录存在"
    ls -la scripts/
else
    echo "❌ scripts目录不存在"
fi

echo -e "\n4. 查找可能的训练脚本..."
find . -name "*train*.py" -type f

echo -e "\n5. 查找可能的数据文件..."
find . -name "*.bin" -o -name "*.parquet" -o -name "*.json" | head -10

echo -e "\n6. 检查Python路径..."
PYTHONPATH=/mnt/linxid615/bza/nanochat-npu:$PYTHONPATH python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

try:
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    print(f'✅ Base目录: {base_dir}')
    
    import os
    tokens_dir = os.path.join(base_dir, 'tokenized_data')
    print(f'✅ Tokens目录: {tokens_dir}')
    print(f'✅ Tokens目录存在: {os.path.exists(tokens_dir)}')
    
    if os.path.exists(tokens_dir):
        files = os.listdir(tokens_dir)
        print(f'✅ Tokens文件数量: {len(files)}')
        for f in files[:5]:
            print(f'  - {f}')
    
except Exception as e:
    print(f'❌ 错误: {e}')
"

echo -e "\n=== 数据检查完成 ==="