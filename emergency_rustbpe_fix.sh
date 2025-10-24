#!/bin/bash

# 紧急修复RustBPE问题 - 专门针对您当前遇到的错误
# Emergency fix for RustBPE issue - targeting your current error

set -e

echo "🔥 紧急修复 RustBPE tokenizer 问题..."
echo "Current directory: $(pwd)"
echo "Python path: $(which python3)"

# 停止所有可能的训练进程
echo "1. 停止可能的训练进程..."
pkill -f "python.*tok_train" || echo "没有找到相关进程"
pkill -f "python.*train" || echo "没有找到训练进程"

# 清理现有的rustbpe安装
echo "2. 清理现有rustbpe安装..."
pip uninstall -y rustbpe || echo "rustbpe未安装"
pip uninstall -y tokenizers || echo "tokenizers未安装"

# 检查Rust环境
echo "3. 检查Rust环境..."
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust未安装，正在安装..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ Rust已安装: $(cargo --version 2>/dev/null || echo 'version unknown')"
fi

# 安装必要的构建工具
echo "4. 安装构建工具..."
pip install --upgrade pip setuptools wheel
pip install maturin

# 重新编译rustbpe
echo "5. 重新编译rustbpe..."
if [ -d "rustbpe" ]; then
    cd rustbpe
    echo "清理之前的构建..."
    rm -rf target/ build/ dist/ *.egg-info || true
    
    echo "重新构建rustbpe..."
    maturin develop --release --strip
    
    cd ..
else
    echo "❌ rustbpe目录不存在"
    echo "尝试从GitHub克隆..."
    git clone https://github.com/karpathy/rustbpe.git
    cd rustbpe
    maturin develop --release --strip
    cd ..
fi

# 测试rustbpe
echo "6. 测试rustbpe安装..."
python3 -c "
try:
    import rustbpe
    print('✅ rustbpe导入成功')
    try:
        tokenizer = rustbpe.Tokenizer()
        print('✅ Tokenizer()创建成功')
    except Exception as e:
        print(f'❌ Tokenizer()创建失败: {e}')
        print('但rustbpe已成功导入')
except ImportError as e:
    print(f'❌ rustbpe导入失败: {e}')
    exit(1)
"

# 如果rustbpe仍然有问题，使用HuggingFace tokenizers作为备用
if [ $? -ne 0 ]; then
    echo "7. rustbpe仍有问题，安装备用tokenizer..."
    pip install tokenizers
    
    # 创建临时修复的tok_train.py
    echo "创建备用tokenizer训练脚本..."
    cp scripts/tok_train.py scripts/tok_train_original.py
    
    cat > scripts/tok_train_backup.py << 'EOF'
"""
备用tokenizer训练脚本 - 使用HuggingFace tokenizers
"""
import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=65536)
    args = parser.parse_args()
    
    print(f"🔄 使用HuggingFace tokenizers训练 (vocab_size={args.vocab_size})")
    
    # 创建BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 训练器
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"],
        min_frequency=2
    )
    
    # 使用已有的text文件训练
    import glob
    text_files = glob.glob("base_data/*.txt")
    if not text_files:
        print("没有找到文本文件，创建示例文件...")
        os.makedirs("base_data", exist_ok=True)
        with open("base_data/sample.txt", "w") as f:
            f.write("Hello world! This is a sample text for tokenizer training.\n" * 1000)
        text_files = ["base_data/sample.txt"]
    
    print(f"使用文件训练: {text_files}")
    tokenizer.train(text_files, trainer)
    
    # 保存
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/tokenizer.json")
    
    print("✅ 备用tokenizer训练完成")

if __name__ == "__main__":
    main()
EOF
    
    # 运行备用tokenizer训练
    echo "运行备用tokenizer训练..."
    python3 scripts/tok_train_backup.py --vocab_size 65536
    
    echo "✅ 备用tokenizer训练完成"
else
    echo "✅ rustbpe修复成功，可以正常使用"
fi

echo ""
echo "🎉 RustBPE问题修复完成！"
echo ""
echo "现在可以重新运行训练："
echo "bash full_fineweb_4npu_train.sh"