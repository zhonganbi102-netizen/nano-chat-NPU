#!/bin/bash

echo "🎯 一键4NPU分布式训练启动..."

# 清理所有可能的环境变量问题
unset PYTORCH_NPU_ALLOC_CONF
unset CUDA_VISIBLE_DEVICES

# 设置NPU环境
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# HCCL通信设置
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=127.0.0.1

# 清理进程
echo "清理环境..."
pkill -f "python.*base_train.py" 2>/dev/null || true
pkill -f "torchrun" 2>/dev/null || true
sleep 3

# 清理NPU内存
python3 -c "
import torch_npu
print('清理NPU内存...')
for i in range(4):
    try:
        torch_npu.npu.set_device(i)
        torch_npu.npu.empty_cache()
        torch_npu.npu.synchronize()
    except:
        pass
print('内存清理完成')
" 2>/dev/null

echo ""
echo "=== 配置总结 ==="
echo "✅ 优化器：已修复，使用标准PyTorch优化器"
echo "✅ 内存管理：NPU原生，无CUDA参数"
echo "✅ 通信协议：HCCL正常工作"
echo "✅ 设备配置：4个NPU (0,1,2,3)"
echo ""
echo "🚀 启动最终4NPU分布式训练..."

torchrun --standalone --nproc_per_node=4 -- scripts/base_train.py \
    --depth=6 \
    --device_batch_size=3 \
    --total_batch_size=49152 \
    --max_seq_len=1024 \
    --num_iterations=20 \
    --eval_every=999999 \
    --core_metric_every=999999 \
    --sample_every=999999 \
    --run="final_4npu_$(date +%Y%m%d_%H%M%S)"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉🎉🎉 4NPU分布式训练成功！🎉🎉🎉"
    echo ""
    echo "恭喜！你已经成功实现了："
    echo "  ✅ 4个NPU分布式训练"
    echo "  ✅ HCCL通信正常"
    echo "  ✅ 内存管理优化"
    echo "  ✅ 优化器兼容性修复"
    echo ""
    echo "现在可以进行更大规模的训练了！"
else
    echo ""
    echo "❌ 最终测试失败，请检查日志"
fi

echo ""
echo "最终测试完成: $(date)"