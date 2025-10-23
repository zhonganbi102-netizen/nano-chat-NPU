#!/bin/bash

# 快速备用解决方案 - 使用HuggingFace tokenizer替代rustbpe
# Quick fallback solution - Use HuggingFace tokenizer instead of rustbpe

set -e

echo "=== RustBPE备用解决方案 ==="
echo "RustBPE Fallback Solution"

# 1. 安装HuggingFace tokenizers作为备用
echo "1. 安装HuggingFace tokenizers..."
pip install tokenizers

# 2. 创建备用tokenizer训练脚本
echo "2. 创建备用tokenizer脚本..."
cat > scripts/tok_train_fallback.py << 'EOF'
"""
备用tokenizer训练脚本 - 使用HuggingFace tokenizers
Fallback tokenizer training script using HuggingFace tokenizers
"""

import os
import sys
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def train_tokenizer_fallback(vocab_size=65536):
    """使用HuggingFace tokenizers训练BPE tokenizer"""
    print(f"🔄 使用HuggingFace tokenizers训练备用tokenizer (vocab_size={vocab_size})")
    
    # 创建BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 设置特殊token
    special_tokens = ["<unk>", "<s>", "</s>"]
    
    # 创建trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    # 从数据文件训练
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "fineweb")
    
    # 查找parquet文件
    import glob
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"没有在 {data_dir} 中找到parquet文件")
    
    print(f"找到 {len(parquet_files)} 个数据文件")
    
    # 创建文本迭代器
    def text_iterator():
        import pandas as pd
        count = 0
        for file in parquet_files[:5]:  # 只使用前5个文件快速训练
            print(f"处理文件: {file}")
            df = pd.read_parquet(file)
            for text in df['text']:
                yield text
                count += 1
                if count >= 100000:  # 限制训练样本数量
                    return
    
    # 训练tokenizer
    print("开始训练tokenizer...")
    tokenizer.train_from_iterator(text_iterator(), trainer)
    
    # 设置后处理器
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ]
    )
    
    # 保存tokenizer
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    print(f"✅ 备用tokenizer训练完成，保存到: {tokenizer_path}")
    print(f"词汇表大小: {tokenizer.get_vocab_size()}")
    
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=65536)
    args = parser.parse_args()
    
    train_tokenizer_fallback(args.vocab_size)
EOF

# 3. 修改tokenizer.py以支持备用方案
echo "3. 创建tokenizer备用补丁..."
cat > tokenizer_fallback_patch.py << 'EOF'
"""
为tokenizer.py添加备用支持的补丁
"""

# 在nanochat/tokenizer.py中添加备用import
fallback_import = '''
# 备用tokenizer支持
try:
    import rustbpe
    RUSTBPE_AVAILABLE = True
except ImportError:
    RUSTBPE_AVAILABLE = False
    try:
        from tokenizers import Tokenizer
        HUGGINGFACE_TOKENIZERS_AVAILABLE = True
    except ImportError:
        HUGGINGFACE_TOKENIZERS_AVAILABLE = False
'''

print("添加以下代码到nanochat/tokenizer.py的顶部:")
print(fallback_import)
EOF

echo "✅ 备用解决方案准备完成"
echo ""
echo "现在请选择以下方案之一:"
echo "方案1: 尝试修复rustbpe"
echo "  ./fix_rustbpe_server.sh"
echo ""
echo "方案2: 使用备用tokenizer"
echo "  python scripts/tok_train_fallback.py"
echo ""
echo "方案3: 跳过tokenizer训练（使用预训练的）"
echo "  # 直接运行训练，会使用默认tokenizer"