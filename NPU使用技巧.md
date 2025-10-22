# NPU使用技巧和最佳实践

## 🎯 核心优化策略

### 1. 内存管理最佳实践

```python
import torch_npu

# 在训练开始前清理内存
torch_npu.npu.empty_cache()

# 监控内存使用
def print_memory_usage():
    allocated = torch_npu.npu.memory_allocated() / 1024**3
    cached = torch_npu.npu.memory_reserved() / 1024**3
    print(f"内存: 已分配 {allocated:.2f}GB, 缓存 {cached:.2f}GB")

# 在关键点监控内存
print_memory_usage()  # 训练前
# ... 训练代码 ...
print_memory_usage()  # 训练后
```

### 2. 批次大小动态调整

```python
# 根据NPU内存动态调整批次大小
def find_optimal_batch_size(model, max_batch_size=32):
    """自动找到最优批次大小"""
    for batch_size in range(max_batch_size, 0, -1):
        try:
            # 测试前向传播
            dummy_input = torch.randint(0, 1000, (batch_size, 512)).to('npu')
            with torch.no_grad():
                output = model(dummy_input)
            print(f"最优批次大小: {batch_size}")
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch_npu.npu.empty_cache()
                continue
            else:
                raise e
    return 1

# 使用示例
optimal_batch_size = find_optimal_batch_size(model)
```

### 3. 梯度累积策略

```python
# 智能梯度累积
def smart_gradient_accumulation(target_batch_size, device_batch_size):
    """计算最优梯度累积步数"""
    accumulation_steps = target_batch_size // device_batch_size
    effective_batch_size = device_batch_size * accumulation_steps
    
    print(f"目标批次: {target_batch_size}")
    print(f"设备批次: {device_batch_size}")
    print(f"累积步数: {accumulation_steps}")
    print(f"有效批次: {effective_batch_size}")
    
    return accumulation_steps

# 使用示例
accumulation_steps = smart_gradient_accumulation(
    target_batch_size=256,
    device_batch_size=32
)
```

## ⚡ 性能优化技巧

### 1. 数据加载优化

```python
# 优化的数据加载器配置
def get_optimized_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,          # NPU推荐8个worker
        pin_memory=True,        # 启用内存锁定
        persistent_workers=True, # 保持worker进程
        prefetch_factor=4,      # 预取因子
    )
```

### 2. 模型编译优化

```python
# NPU特定的编译优化
def compile_model_for_npu(model):
    """为NPU优化模型编译"""
    # 启用torch.compile (如果支持)
    try:
        compiled_model = torch.compile(
            model, 
            mode="reduce-overhead",  # NPU推荐模式
            dynamic=False           # 静态图优化
        )
        print("✅ 模型编译成功")
        return compiled_model
    except Exception as e:
        print(f"⚠️ 编译失败，使用原始模型: {e}")
        return model

# 使用示例
model = compile_model_for_npu(model)
```

### 3. 自动混合精度优化

```python
# NPU混合精度最佳实践
def setup_mixed_precision():
    """配置NPU混合精度训练"""
    scaler = torch.npu.amp.GradScaler()
    
    # 训练循环中的使用
    def training_step(model, data, optimizer, scaler):
        with torch.npu.amp.autocast():
            output = model(data)
            loss = compute_loss(output)
        
        # 梯度缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        return loss.item()
    
    return scaler, training_step
```

## 🔧 调试技巧

### 1. NPU状态检查工具

```python
def npu_health_check():
    """NPU健康状态检查"""
    import torch_npu
    
    print("=== NPU状态检查 ===")
    
    # 基本信息
    print(f"NPU可用: {torch_npu.npu.is_available()}")
    print(f"NPU数量: {torch_npu.npu.device_count()}")
    print(f"当前设备: {torch_npu.npu.current_device()}")
    
    # 内存信息
    for i in range(torch_npu.npu.device_count()):
        allocated = torch_npu.npu.memory_allocated(i) / 1024**3
        cached = torch_npu.npu.memory_reserved(i) / 1024**3
        print(f"NPU {i}: 已用 {allocated:.2f}GB, 缓存 {cached:.2f}GB")
    
    # 温度和功耗 (如果支持)
    try:
        import subprocess
        result = subprocess.run(['npu-smi', 'info'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("NPU设备信息:")
            print(result.stdout)
    except:
        print("无法获取NPU设备详细信息")

# 在训练开始前运行
npu_health_check()
```

### 2. 错误诊断工具

```python
def diagnose_npu_error(error_msg):
    """NPU错误诊断助手"""
    diagnostics = {
        "out of memory": [
            "减小batch_size",
            "增加gradient_accumulation_steps", 
            "使用gradient_checkpointing",
            "清理NPU缓存: torch_npu.npu.empty_cache()"
        ],
        "HCCL": [
            "检查网络连通性",
            "验证MASTER_ADDR和MASTER_PORT",
            "确认WORLD_SIZE和RANK设置",
            "重启分布式训练"
        ],
        "compile": [
            "检查CANN版本兼容性",
            "验证torch_npu版本",
            "尝试disable torch.compile",
            "检查算子支持情况"
        ]
    }
    
    print(f"错误诊断: {error_msg}")
    for keyword, solutions in diagnostics.items():
        if keyword in error_msg.lower():
            print(f"\n可能的解决方案 ({keyword}):")
            for i, solution in enumerate(solutions, 1):
                print(f"{i}. {solution}")
            break
    else:
        print("未找到匹配的解决方案，请查看详细日志")

# 使用示例
try:
    # 训练代码
    pass
except Exception as e:
    diagnose_npu_error(str(e))
```

## 📊 性能监控工具

### 1. 实时性能监控

```python
import time
import threading
from collections import deque

class NPUMonitor:
    """NPU性能实时监控"""
    
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.metrics = {
            'memory_used': deque(maxlen=100),
            'memory_cached': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        print("🔍 NPU监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        print("⏹️ NPU监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        import torch_npu
        
        while self.running:
            try:
                memory_used = torch_npu.npu.memory_allocated() / 1024**3
                memory_cached = torch_npu.npu.memory_reserved() / 1024**3
                timestamp = time.time()
                
                self.metrics['memory_used'].append(memory_used)
                self.metrics['memory_cached'].append(memory_cached)
                self.metrics['timestamps'].append(timestamp)
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def get_summary(self):
        """获取监控摘要"""
        if not self.metrics['memory_used']:
            return "无监控数据"
        
        avg_memory = sum(self.metrics['memory_used']) / len(self.metrics['memory_used'])
        max_memory = max(self.metrics['memory_used'])
        
        return f"""
NPU监控摘要:
- 平均内存使用: {avg_memory:.2f} GB
- 峰值内存使用: {max_memory:.2f} GB
- 监控时长: {len(self.metrics['memory_used'])} 秒
        """

# 使用示例
monitor = NPUMonitor()
monitor.start_monitoring()

# 训练代码
try:
    # ... 训练循环 ...
    pass
finally:
    monitor.stop_monitoring()
    print(monitor.get_summary())
```

### 2. 训练速度分析

```python
class TrainingProfiler:
    """训练性能分析器"""
    
    def __init__(self):
        self.step_times = []
        self.throughput_history = []
    
    def start_step(self):
        """开始计时一个训练步"""
        self.step_start = time.time()
    
    def end_step(self, num_tokens):
        """结束计时并记录吞吐量"""
        step_time = time.time() - self.step_start
        throughput = num_tokens / step_time
        
        self.step_times.append(step_time)
        self.throughput_history.append(throughput)
        
        return step_time, throughput
    
    def get_stats(self):
        """获取性能统计"""
        if not self.step_times:
            return "无性能数据"
        
        avg_time = sum(self.step_times) / len(self.step_times)
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        return f"""
训练性能统计:
- 平均步时间: {avg_time:.3f} 秒
- 平均吞吐量: {avg_throughput:.0f} tokens/秒
- 总步数: {len(self.step_times)}
        """

# 使用示例
profiler = TrainingProfiler()

for step in range(num_steps):
    profiler.start_step()
    
    # 训练代码
    loss = training_step()
    
    step_time, throughput = profiler.end_step(num_tokens=batch_size * seq_len)
    
    if step % 10 == 0:
        print(f"Step {step}: {step_time:.3f}s, {throughput:.0f} tok/s")

print(profiler.get_stats())
```

## 🚀 生产环境优化

### 1. 服务稳定性

```python
import functools
import logging

def npu_error_retry(max_retries=3, delay=1.0):
    """NPU错误重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "NPU" in str(e) and attempt < max_retries - 1:
                        logging.warning(f"NPU错误，重试 {attempt + 1}/{max_retries}: {e}")
                        torch_npu.npu.empty_cache()
                        time.sleep(delay * (2 ** attempt))  # 指数退避
                        continue
                    raise
            return None
        return wrapper
    return decorator

# 使用示例
@npu_error_retry(max_retries=3)
def inference_with_retry(model, input_text):
    """带重试的推理函数"""
    tokens = tokenizer.encode(input_text)
    with torch.no_grad():
        output = model.generate(tokens)
    return tokenizer.decode(output)
```

### 2. 资源管理

```python
class NPUResourceManager:
    """NPU资源管理器"""
    
    def __init__(self):
        self.models = {}
        self.memory_threshold = 0.9  # 90%内存使用率阈值
    
    def load_model(self, model_name, model_path):
        """智能模型加载"""
        # 检查内存是否足够
        if self._check_memory_pressure():
            self._cleanup_unused_models()
        
        # 加载模型
        model = load_model_from_path(model_path)
        model = model.to('npu')
        self.models[model_name] = {
            'model': model,
            'last_used': time.time(),
            'usage_count': 0
        }
        
        return model
    
    def get_model(self, model_name):
        """获取模型并更新使用统计"""
        if model_name in self.models:
            self.models[model_name]['last_used'] = time.time()
            self.models[model_name]['usage_count'] += 1
            return self.models[model_name]['model']
        return None
    
    def _check_memory_pressure(self):
        """检查内存压力"""
        import torch_npu
        allocated = torch_npu.npu.memory_allocated()
        total = torch_npu.npu.get_device_properties(0).total_memory
        return allocated / total > self.memory_threshold
    
    def _cleanup_unused_models(self):
        """清理未使用的模型"""
        current_time = time.time()
        to_remove = []
        
        for name, info in self.models.items():
            # 10分钟未使用的模型
            if current_time - info['last_used'] > 600:
                to_remove.append(name)
        
        for name in to_remove:
            del self.models[name]['model']
            del self.models[name]
            torch_npu.npu.empty_cache()
            print(f"清理模型: {name}")

# 使用示例
resource_manager = NPUResourceManager()
model = resource_manager.load_model("chat_model", "/path/to/model")
```

这些技巧和工具可以帮助你在华为昇腾NPU上更好地运行nanochat，提升训练和推理的效率与稳定性。记住要根据具体的硬件配置和应用场景进行调整！
