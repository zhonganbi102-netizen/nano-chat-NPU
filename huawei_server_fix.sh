#!/bin/bash

# 华为服务器专用 - 无curl环境的rustbpe修复脚本
# Huawei server specific - rustbpe fix without curl

set -e

echo "🔥 华为服务器专用 - RustBPE修复脚本"
echo "当前目录: $(pwd)"
echo "Python路径: $(which python3)"

# 1. 检查系统工具
echo "1. 检查系统环境..."
echo "系统信息: $(uname -a)"

# 检查包管理器
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum" 
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
else
    PKG_MANAGER="none"
fi

echo "包管理器: $PKG_MANAGER"

# 2. 安装curl（如果需要）
if ! command -v curl &> /dev/null; then
    echo "2. 安装curl..."
    if [ "$PKG_MANAGER" = "apt-get" ]; then
        apt-get update && apt-get install -y curl
    elif [ "$PKG_MANAGER" = "yum" ]; then
        yum install -y curl
    elif [ "$PKG_MANAGER" = "dnf" ]; then
        dnf install -y curl
    else
        echo "❌ 无法自动安装curl，请手动安装"
        echo "使用跳过tokenizer的方案..."
        bash skip_tokenizer_train.sh
        exit 0
    fi
else
    echo "2. ✅ curl已安装"
fi

# 3. 安装Rust（使用wget作为备用）
echo "3. 安装Rust环境..."
if ! command -v cargo &> /dev/null; then
    echo "正在安装Rust..."
    if command -v curl &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    elif command -v wget &> /dev/null; then
        wget -qO- https://sh.rustup.rs | sh -s -- -y
    else
        echo "❌ 既没有curl也没有wget，无法安装Rust"
        echo "使用跳过tokenizer的方案..."
        bash skip_tokenizer_train.sh
        exit 0
    fi
    
    # 加载Rust环境
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ Rust安装完成: $(cargo --version)"
else
    echo "✅ Rust已安装: $(cargo --version)"
fi

# 4. 清理并重新安装rustbpe
echo "4. 重新编译rustbpe..."

# 停止相关进程
pkill -f "python.*tok_train" || echo "没有相关进程"

# 清理现有安装
pip uninstall -y rustbpe || echo "rustbpe未安装"

# 安装编译工具
echo "安装编译依赖..."
pip install --upgrade pip setuptools wheel maturin --root-user-action=ignore

# 检查rustbpe目录
if [ -d "rustbpe" ]; then
    echo "清理rustbpe构建缓存..."
    cd rustbpe
    rm -rf target/ build/ dist/ *.egg-info || true
    
    echo "重新编译rustbpe..."
    # 使用build + pip install方式，而不是develop
    maturin build --release
    pip install target/wheels/*.whl --force-reinstall --root-user-action=ignore
    cd ..
else
    echo "❌ rustbpe目录不存在，从GitHub克隆..."
    if command -v git &> /dev/null; then
        git clone https://github.com/karpathy/rustbpe.git
        cd rustbpe
        maturin build --release
        pip install target/wheels/*.whl --force-reinstall --root-user-action=ignore
        cd ..
    else
        echo "❌ git不可用，无法克隆rustbpe"
        echo "使用跳过tokenizer的方案..."
        bash skip_tokenizer_train.sh
        exit 0
    fi
fi

# 5. 测试rustbpe
echo "5. 测试rustbpe安装..."
python3 -c "
import sys
try:
    import rustbpe
    print('✅ rustbpe导入成功')
    try:
        tokenizer = rustbpe.Tokenizer()
        print('✅ Tokenizer创建成功')
        print('🎉 rustbpe完全修复成功！')
        sys.exit(0)
    except Exception as e:
        print(f'⚠️  Tokenizer创建失败: {e}')
        print('但rustbpe导入成功，可能是接口问题')
        sys.exit(1)
except ImportError as e:
    print(f'❌ rustbpe导入失败: {e}')
    sys.exit(2)
"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ rustbpe修复成功！"
    echo "现在可以正常运行训练："
    echo "bash full_fineweb_4npu_train.sh"
elif [ $exit_code -eq 1 ]; then
    echo "⚠️  rustbpe部分成功，但Tokenizer接口有问题"
    echo "使用HuggingFace tokenizer备用方案..."
    pip install tokenizers --root-user-action=ignore
    bash rustbpe_fallback_solution.sh
elif [ $exit_code -eq 2 ]; then
    echo "❌ rustbpe安装失败，使用跳过方案..."
    bash skip_tokenizer_train.sh
fi

echo ""
echo "🎉 修复流程完成！"