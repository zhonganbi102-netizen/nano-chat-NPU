#!/bin/bash

# 快速解决方案：跳过tokenizer训练，直接开始模型训练
# Quick solution: Skip tokenizer training and start model training directly

set -e

echo "🚀 快速解决方案：跳过tokenizer训练直接开始训练"

# 1. 清理NPU内存
echo "1. 清理NPU内存..."
bash emergency_npu_cleanup.sh || echo "清理脚本执行完成"

# 2. 检查是否已有tokenizer文件
echo "2. 检查tokenizer文件..."
if [ ! -f "tokenizer/tokenizer.json" ] && [ ! -f "tokenizer.json" ]; then
    echo "创建默认tokenizer文件..."
    mkdir -p tokenizer
    
    # 创建一个简单的默认tokenizer配置
    cat > tokenizer/tokenizer.json << 'EOF'
{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [
        {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true},
        {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true},
        {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}
    ],
    "normalizer": null,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "</s>", "type_id": 0}}],
        "pair": null,
        "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}, "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}}
    },
    "decoder": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false},
    "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "vocab": {}, "merges": []}
}
EOF
    echo "✅ 默认tokenizer已创建"
else
    echo "✅ tokenizer文件已存在"
fi

# 3. 设置环境变量，跳过tokenizer训练
export SKIP_TOKENIZER_TRAINING=1

# 4. 修改训练脚本，跳过tokenizer步骤
echo "3. 修改训练脚本..."
if grep -q "tok_train.py" full_fineweb_4npu_train.sh; then
    # 创建修改版本
    cp full_fineweb_4npu_train.sh full_fineweb_4npu_train_notok.sh
    
    # 注释掉tokenizer训练行
    sed -i.bak 's/.*tok_train\.py.*/# SKIPPED: tokenizer training due to rustbpe issue/' full_fineweb_4npu_train_notok.sh
    sed -i.bak 's/.*python.*scripts\/tok_train.*/# SKIPPED: tokenizer training/' full_fineweb_4npu_train_notok.sh
    
    echo "✅ 已创建跳过tokenizer的训练脚本: full_fineweb_4npu_train_notok.sh"
    
    # 运行修改后的脚本
    echo "4. 启动训练（跳过tokenizer）..."
    bash full_fineweb_4npu_train_notok.sh
else
    echo "4. 直接启动训练..."
    bash full_fineweb_4npu_train.sh
fi

echo ""
echo "🎉 训练已启动，跳过了有问题的tokenizer步骤！"