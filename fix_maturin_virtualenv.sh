#!/bin/bash

echo "=== RustBPE Maturin虚拟环境问题修复 ==="

# 确保在rustbpe目录
if [ ! -f "Cargo.toml" ]; then
    echo "❌ 请在rustbpe目录中运行此脚本"
    exit 1
fi

# 加载Rust环境
source ~/.cargo/env 2>/dev/null || true
export PATH="$HOME/.cargo/bin:$PATH"

echo "当前目录: $(pwd)"
echo "Python路径: $(which python)"
echo "pip路径: $(which pip)"

# 方法1: 使用maturin build + pip install
echo ""
echo "=== 方法1: 构建wheel文件 ==="
echo "运行: maturin build --release"

if maturin build --release; then
    echo "✅ wheel构建成功"
    
    # 查找wheel文件
    if [ -d "target/wheels" ]; then
        WHEEL_FILES=(target/wheels/*.whl)
        if [ -e "${WHEEL_FILES[0]}" ]; then
            WHEEL_FILE="${WHEEL_FILES[0]}"
            echo "找到wheel文件: $WHEEL_FILE"
            
            # 安装wheel
            echo "安装wheel到Python环境..."
            if pip install "$WHEEL_FILE" --force-reinstall --no-deps; then
                echo "✅ wheel安装成功"
                SUCCESS=true
            else
                echo "❌ wheel安装失败"
            fi
        else
            echo "❌ 找不到wheel文件"
        fi
    else
        echo "❌ target/wheels目录不存在"
    fi
else
    echo "❌ wheel构建失败"
fi

# 如果方法1失败，尝试方法2
if [ "$SUCCESS" != "true" ]; then
    echo ""
    echo "=== 方法2: 设置虚拟环境变量 ==="
    
    # 创建假的虚拟环境变量
    export VIRTUAL_ENV="/usr/local/python3.11.13"
    export CONDA_PREFIX="/usr/local/python3.11.13"
    
    echo "设置虚拟环境变量:"
    echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
    echo "  CONDA_PREFIX=$CONDA_PREFIX"
    
    if maturin develop --release; then
        echo "✅ maturin develop成功"
        SUCCESS=true
    else
        echo "❌ maturin develop仍然失败"
    fi
fi

# 如果方法2失败，尝试方法3
if [ "$SUCCESS" != "true" ]; then
    echo ""
    echo "=== 方法3: 手动编译安装 ==="
    
    # 确保有setup.py或pyproject.toml
    if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
        echo "创建基本的setup.py..."
        cat > setup.py << 'EOF'
from setuptools import setup
from pyo3_setuptools_rust import Pyo3RustExtension, build_rust

setup(
    name="rustbpe",
    rust_extensions=[Pyo3RustExtension("rustbpe.rustbpe", "Cargo.toml")],
    packages=["rustbpe"],
    zip_safe=False,
    cmdclass={"build_rust": build_rust}
)
EOF
    fi
    
    # 尝试直接pip安装
    if pip install . --force-reinstall --no-build-isolation; then
        echo "✅ 直接pip安装成功"
        SUCCESS=true
    else
        echo "❌ 直接pip安装失败"
    fi
fi

# 验证安装
echo ""
echo "=== 验证安装 ==="
cd ..

python -c "
try:
    import rustbpe
    print('✅ rustbpe导入成功')
    
    # 检查属性
    attrs = [attr for attr in dir(rustbpe) if not attr.startswith('_')]
    print(f'可用属性: {attrs}')
    
    # 测试Tokenizer
    if hasattr(rustbpe, 'Tokenizer'):
        tok = rustbpe.Tokenizer()
        print('✅ Tokenizer类可用')
        
        # 编码测试
        result = tok.encode('Test')
        print(f'✅ 编码测试成功: {result}')
        
        print('')
        print('🎉 rustbpe安装验证成功！')
        print('现在可以运行: ./speedrun_npu.sh')
        
    else:
        print('❌ Tokenizer类不可用')
        print('可用属性:', attrs)
        
except Exception as e:
    print(f'❌ 验证失败: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "=== 修复完成 ==="