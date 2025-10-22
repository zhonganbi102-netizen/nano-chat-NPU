#!/bin/bash

echo "=== 检查NanoChat模块结构 ==="

cd /mnt/linxid615/bza/nanochat-npu

echo "1. 当前目录内容:"
ls -la

echo ""
echo "2. 检查nanochat目录:"
if [ -d "nanochat" ]; then
    echo "✅ nanochat目录存在"
    echo "nanochat目录内容:"
    ls -la nanochat/
else
    echo "❌ nanochat目录不存在"
fi

echo ""
echo "3. 检查Python包结构:"
if [ -f "nanochat/__init__.py" ]; then
    echo "✅ nanochat/__init__.py 存在"
else
    echo "❌ nanochat/__init__.py 不存在"
fi

if [ -f "nanochat/model.py" ]; then
    echo "✅ nanochat/model.py 存在"
else
    echo "❌ nanochat/model.py 不存在"
fi

echo ""
echo "4. 查找所有Python文件:"
find . -name "*.py" -type f | head -10

echo ""
echo "5. 查找model相关文件:"
find . -name "*model*" -type f

echo ""
echo "6. 检查scripts目录:"
if [ -d "scripts" ]; then
    echo "✅ scripts目录存在"
    ls -la scripts/ | head -5
else
    echo "❌ scripts目录不存在"
fi

echo ""
echo "7. 尝试Python导入测试:"
python3 -c "
import sys
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')
print('Python路径:', sys.path[:3])

try:
    import nanochat
    print('✅ nanochat包导入成功')
    print('nanochat模块位置:', nanochat.__file__)
    print('nanochat模块内容:', dir(nanochat))
except Exception as e:
    print('❌ nanochat包导入失败:', e)

try:
    import os
    if os.path.exists('nanochat'):
        print('✅ nanochat目录存在于Python中')
        files = os.listdir('nanochat')
        print('nanochat文件:', files[:5])
    else:
        print('❌ nanochat目录在Python中不可见')
except Exception as e:
    print('检查目录时出错:', e)
"