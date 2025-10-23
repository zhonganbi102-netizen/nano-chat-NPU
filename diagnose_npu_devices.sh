#!/bin/bash

# NPU设备诊断脚本 - 解决Device:-1问题

echo "🔍 NPU设备诊断 - 解决Device:-1问题"
echo "=================================================="

# 1. 检查NPU硬件状态
echo "1. 检查NPU硬件状态..."
echo ""
npu-smi info
echo ""

# 2. 检查NPU设备数量
echo "2. 检查NPU设备映射..."
if command -v npu-smi >/dev/null 2>&1; then
    DEVICE_COUNT=$(npu-smi info | grep -E "^\| [0-9]+" | wc -l)
    echo "✅ 硬件NPU设备数量: $DEVICE_COUNT"
    
    # 显示设备ID
    echo "NPU设备ID列表:"
    npu-smi info | grep -E "^\| [0-9]+" | awk '{print "  NPU " $2}'
else
    echo "❌ npu-smi命令不可用"
fi

echo ""

# 3. 设置环境并测试torch_npu
echo "3. 测试torch_npu设备检测..."

# 动态查找环境
if [ -f ".ascend_env_path" ]; then
    source .ascend_env_path
    source "$ASCEND_SET_ENV_PATH"
    echo "✅ 使用环境文件: $ASCEND_SET_ENV_PATH"
else
    # 尝试常见路径
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        echo "✅ 使用标准环境文件"
    else
        echo "⚠️ 未找到set_env.sh，手动设置环境"
        export ASCEND_HOME="/usr/local/Ascend/ascend-toolkit"
        export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
        export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH"
    fi
fi

echo ""

# 4. Python级别的NPU检测
echo "4. Python级别NPU设备检测..."
python3 << 'EOF'
import sys
print(f"Python版本: {sys.version}")
print("")

try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    
    # 检测NPU
    try:
        import torch_npu
        print(f"✅ torch_npu已导入: {torch_npu.__version__}")
        
        # 检查NPU设备数量
        if hasattr(torch_npu, 'npu') and hasattr(torch_npu.npu, 'device_count'):
            npu_count = torch_npu.npu.device_count()
            print(f"✅ torch_npu检测到设备数: {npu_count}")
            
            if npu_count > 0:
                for i in range(npu_count):
                    try:
                        device_name = torch_npu.npu.get_device_name(i)
                        print(f"  NPU {i}: {device_name}")
                    except Exception as e:
                        print(f"  NPU {i}: 无法获取设备名 ({e})")
            else:
                print("❌ torch_npu未检测到任何NPU设备！")
        else:
            print("❌ torch_npu.npu.device_count()不可用")
            
        # 测试设备设置
        try:
            torch_npu.npu.set_device(0)
            print("✅ 成功设置NPU设备0")
        except Exception as e:
            print(f"❌ 设置NPU设备0失败: {e}")
            
    except ImportError as e:
        print(f"❌ 无法导入torch_npu: {e}")
    except Exception as e:
        print(f"❌ torch_npu错误: {e}")
        
except ImportError:
    print("❌ 无法导入PyTorch")
except Exception as e:
    print(f"❌ PyTorch错误: {e}")

print("")

# 检查TBE模块
try:
    import tbe
    print("✅ TBE模块可用")
except ImportError:
    print("❌ TBE模块不可用")
except Exception as e:
    print(f"❌ TBE模块错误: {e}")

EOF

echo ""

# 5. 检查NPU驱动状态
echo "5. 检查NPU驱动状态..."
if [ -f "/proc/driver/davinci_dev" ]; then
    echo "✅ NPU驱动已加载"
    cat /proc/driver/davinci_dev 2>/dev/null || echo "无法读取驱动详细信息"
else
    echo "❌ NPU驱动未找到 (/proc/driver/davinci_dev)"
fi

echo ""

# 6. 环境变量检查
echo "6. 关键环境变量检查..."
echo "ASCEND_HOME: ${ASCEND_HOME:-'未设置'}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-'未设置'}"
echo "PYTHONPATH: ${PYTHONPATH:-'未设置'}"
echo "WORLD_SIZE: ${WORLD_SIZE:-'未设置'}"

echo ""

# 7. 建议
echo "🔧 诊断建议..."
echo "=================================================="

if [ "$DEVICE_COUNT" -eq 0 ] || [ -z "$DEVICE_COUNT" ]; then
    echo "❌ 硬件层面NPU设备未检测到"
    echo "   建议: 重启NPU驱动或检查硬件连接"
    echo "   命令: /usr/local/Ascend/driver/tools/docker_start.sh"
elif python3 -c "import torch_npu; print('ok')" 2>/dev/null; then
    echo "✅ torch_npu可用，问题可能在环境继承"
    echo "   建议: 使用单NPU训练避开分布式问题"
else
    echo "❌ torch_npu不可用"
    echo "   建议: 检查环境变量和安装"
fi

echo ""
echo "🎯 下一步操作建议:"
echo "1. 如果NPU硬件正常: ./single_npu_fineweb_train.sh"
echo "2. 如果需要重启驱动: /usr/local/Ascend/driver/tools/docker_start.sh"
echo "3. 如果环境有问题: 重新设置环境变量"
