# 4张NPU部署指南

## 快速开始

### 1. 克隆项目并安装依赖
```bash
git clone https://github.com/bizhongan/nanochat-npu.git
cd nanochat-npu
chmod +x install_dependencies.sh build_rustbpe.sh speedrun_npu.sh
./install_dependencies.sh
```

### 2. 构建tokenizer
```bash
./build_rustbpe.sh
```

### 3. 验证NPU环境
```bash
python -c "
import torch_npu
print(f'NPU设备数量: {torch_npu.npu.device_count()}')
for i in range(torch_npu.npu.device_count()):
    print(f'NPU {i}: {torch_npu.npu.get_device_name(i)}')
"
```

### 4. 运行4张NPU训练
```bash
./speedrun_npu.sh
```

## 4张NPU配置说明

当前配置已优化为4张NPU：

- **WORLD_SIZE**: 4（分布式训练进程数）
- **ASCEND_RT_VISIBLE_DEVICES**: 0,1,2,3（使用前4张NPU）

### 批次大小优化
- **Base Training**: device_batch_size=20 (比5张NPU时的16更大)
- **Mid Training**: device_batch_size=20  
- **SFT**: device_batch_size=10 (比5张NPU时的8更大)
- **RL**: device_batch_size=5 (比5张NPU时的4更大)

## 预期性能

4张Ascend 910B NPU配置：
- 总显存: ~256GB (64GB × 4)
- 并行效率: 优于5张NPU配置（更均匀的负载分布）
- 训练速度: 预计比单张NPU快3.2-3.6倍

## 故障排除

### 常见问题
1. **找不到NPU设备**
   ```bash
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
   npu-smi info
   ```

2. **tokenizer编译失败**
   ```bash
   ./build_rustbpe.sh
   # 或手动安装Rust和maturin
   ```

3. **内存不足**
   - 减少device_batch_size
   - 确保没有其他进程占用NPU显存

### 验证命令
```bash
# 检查NPU状态
npu-smi info

# 测试基本NPU操作
python -c "
import torch
import torch_npu
torch.npu.set_device(0)
x = torch.randn(1000, 1000).npu()
print('NPU基本操作正常')
"

# 验证分布式后端
python -c "
import torch.distributed as dist
print(f'HCCL可用: {dist.is_hccl_available()}')
"
```

## 自动模式

如果你的NPU数量可能变化，使用自动检测脚本：
```bash
./speedrun_npu_simple.sh
```

这会自动检测NPU数量并调整训练参数。