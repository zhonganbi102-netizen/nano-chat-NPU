#!/usr/bin/env python3
"""
4NPU分布式环境检查脚本
在运行分布式训练前检查环境是否正确配置
"""

import os
import sys
import subprocess
import torch

def run_command(cmd):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "命令超时"
    except Exception as e:
        return False, "", str(e)

def check_npu_hardware():
    """检查NPU硬件状态"""
    print("🔍 检查NPU硬件状态...")
    
    success, stdout, stderr = run_command("npu-smi info")
    if not success:
        print("❌ npu-smi命令失败")
        return False
    
    # 检查是否有4个NPU可用
    lines = stdout.split('\n')
    npu_count = 0
    for line in lines:
        if '910B3' in line and 'OK' in line:
            npu_count += 1
    
    if npu_count >= 4:
        print(f"✅ 检测到 {npu_count} 个可用NPU")
        return True
    else:
        print(f"❌ 只检测到 {npu_count} 个NPU，需要至少4个")
        return False

def check_torch_npu():
    """检查torch_npu环境"""
    print("🔍 检查torch_npu环境...")
    
    try:
        import torch_npu
        print(f"✅ torch_npu版本: {torch_npu.__version__}")
        
        if torch_npu.npu.is_available():
            device_count = torch_npu.npu.device_count()
            print(f"✅ NPU可用，设备数量: {device_count}")
            
            if device_count >= 4:
                print("✅ NPU设备数量足够进行4卡训练")
                return True
            else:
                print("❌ NPU设备数量不足，无法进行4卡训练")
                return False
        else:
            print("❌ NPU不可用")
            return False
            
    except ImportError:
        print("❌ torch_npu未安装")
        return False
    except Exception as e:
        print(f"❌ torch_npu错误: {e}")
        return False

def check_distributed_env():
    """检查分布式环境变量"""
    print("🔍 检查分布式环境变量...")
    
    required_vars = {
        'WORLD_SIZE': '4',
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': '29500'
    }
    
    all_good = True
    for var, expected in required_vars.items():
        value = os.environ.get(var)
        if value == expected:
            print(f"✅ {var}={value}")
        else:
            print(f"❌ {var}={value} (期望: {expected})")
            all_good = False
    
    return all_good

def test_simple_npu_ops():
    """测试简单的NPU操作"""
    print("🔍 测试NPU基本操作...")
    
    try:
        import torch_npu
        
        # 测试每个NPU设备
        for i in range(min(4, torch_npu.npu.device_count())):
            device = torch.device(f'npu:{i}')
            
            # 创建测试张量
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.matmul(test_tensor, test_tensor.t())
            
            print(f"✅ NPU {i} 基本操作测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ NPU操作测试失败: {e}")
        return False

def test_hccl_setup():
    """测试HCCL通信设置"""
    print("🔍 检查HCCL环境设置...")
    
    hccl_vars = {
        'HCCL_WHITELIST_DISABLE': '1',
        'HCCL_IF_IP': '127.0.0.1'
    }
    
    all_good = True
    for var, expected in hccl_vars.items():
        value = os.environ.get(var)
        if value == expected:
            print(f"✅ {var}={value}")
        else:
            print(f"⚠️  {var}={value} (建议: {expected})")
    
    return True

def check_port_availability():
    """检查端口可用性"""
    print("🔍 检查端口可用性...")
    
    port = int(os.environ.get('MASTER_PORT', '29500'))
    
    success, stdout, stderr = run_command(f"netstat -tuln | grep {port}")
    if success and stdout.strip():
        print(f"⚠️  端口 {port} 可能已被占用")
        print("建议使用不同的MASTER_PORT")
        return False
    else:
        print(f"✅ 端口 {port} 可用")
        return True

def main():
    """主检查函数"""
    print("=" * 60)
    print("4NPU分布式训练环境检查")
    print("=" * 60)
    
    checks = [
        ("NPU硬件状态", check_npu_hardware),
        ("torch_npu环境", check_torch_npu), 
        ("分布式环境变量", check_distributed_env),
        ("NPU基本操作", test_simple_npu_ops),
        ("HCCL环境设置", test_hccl_setup),
        ("端口可用性", check_port_availability)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 检查 {name} 时出错: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("检查结果汇总:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有检查通过！可以开始4NPU分布式训练")
        print("建议运行: ./npu_4gpu_conservative.sh (保守配置)")
        print("或者运行: ./npu_4gpu_train.sh (标准配置)")
    else:
        print("⚠️  部分检查未通过，建议修复后再进行分布式训练")
        print("可以先尝试单NPU训练: ./npu_simple_train.sh")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
