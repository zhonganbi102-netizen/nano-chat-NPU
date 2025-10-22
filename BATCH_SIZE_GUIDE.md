# NPU训练批次大小计算说明

## 关键约束
nanochat要求：`total_batch_size % world_tokens_per_fwdbwd == 0`

其中：
- `world_tokens_per_fwdbwd = device_batch_size * max_seq_len * ddp_world_size`

## 计算示例

### 单NPU (ddp_world_size = 1)
- device_batch_size = 4
- max_seq_len = 512  
- world_tokens_per_fwdbwd = 4 * 512 * 1 = 2048
- total_batch_size 必须是 2048 的倍数
- ✅ 正确: total_batch_size = 2048, 4096, 6144...
- ❌ 错误: total_batch_size = 32, 64, 1024...

### 2NPU (ddp_world_size = 2)  
- device_batch_size = 2
- max_seq_len = 512
- world_tokens_per_fwdbwd = 2 * 512 * 2 = 2048  
- total_batch_size 必须是 2048 的倍数

### 4NPU (ddp_world_size = 4)
- device_batch_size = 2  
- max_seq_len = 512
- world_tokens_per_fwdbwd = 2 * 512 * 4 = 4096
- total_batch_size 必须是 4096 的倍数

## 梯度累积
实际的梯度累积步数 = total_batch_size / world_tokens_per_fwdbwd

- 单NPU: grad_accum_steps = 2048 / 2048 = 1  
- 2NPU: grad_accum_steps = 2048 / 2048 = 1
- 4NPU: grad_accum_steps = 4096 / 4096 = 1

如果想要更多梯度累积，可以增加total_batch_size:
- 4NPU: total_batch_size = 8192 => grad_accum_steps = 2