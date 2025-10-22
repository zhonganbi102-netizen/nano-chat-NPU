#!/usr/bin/env python3
"""
简化的NPU训练脚本，专门为华为昇腾NPU优化
避免设备检测问题，直接使用NPU训练
"""

import os
import sys
import time
import torch

# 强制添加项目路径
sys.path.insert(0, '/mnt/linxid615/bza/nanochat-npu')

def main():
    print("=== 🚀 简化NPU训练脚本 ===")
    
    # 1. 检查NPU环境
    print("1. 检查NPU环境...")
    try:
        import torch_npu
        if not torch_npu.npu.is_available():
            print("❌ NPU不可用")
            return
        
        device_count = torch_npu.npu.device_count()
        print(f"✅ NPU可用，设备数: {device_count}")
        
        # 使用第一个NPU设备
        device = torch.device("npu:0")
        torch_npu.npu.set_device(0)
        
    except ImportError:
        print("❌ torch_npu未安装")
        return
    except Exception as e:
        print(f"❌ NPU初始化失败: {e}")
        return
    
    # 2. 导入nanochat模块
    print("2. 导入模块...")
    try:
        from nanochat.gpt import GPT, GPTConfig
        from nanochat.tokenizer import get_tokenizer
        from nanochat.dataset import list_parquet_files, parquets_iter_batched
        print("✅ 模块导入成功")
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 检查数据
    print("3. 检查数据文件...")
    try:
        files = list_parquet_files()
        if len(files) == 0:
            print("❌ 没有找到数据文件")
            print("请先运行: rm -rf /root/.cache/nanochat/base_data && ln -sf /mnt/linxid615/bza/nanochat-npu/base_data /root/.cache/nanochat/base_data")
            return
        print(f"✅ 找到 {len(files)} 个数据文件")
    except Exception as e:
        print(f"❌ 数据检查失败: {e}")
        return
    
    # 4. 加载tokenizer
    print("4. 加载tokenizer...")
    try:
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        print(f"✅ Tokenizer加载成功: vocab_size={vocab_size}")
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        return
    
    # 5. 创建模型
    print("5. 创建模型...")
    try:
        # 小模型配置
        depth = 8
        max_seq_len = 256
        model_dim = depth * 64
        num_heads = max(1, (model_dim + 127) // 128)
        
        model_config = GPTConfig(
            sequence_len=max_seq_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim
        )
        
        # 在CPU上创建模型，然后移动到NPU
        model = GPT(model_config)
        model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功: {num_params:,} 参数")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. 简化的数据加载
    print("6. 创建数据迭代器...")
    try:
        def simple_data_loader():
            """简化的数据加载器，直接使用NPU"""
            batch_size = 4
            seq_len = 256
            
            for batch in parquets_iter_batched("train", start=0, step=1):
                if len(batch) == 0:
                    continue
                
                # 使用tokenizer编码
                try:
                    tokens_list = tokenizer.encode(batch[:batch_size], prepend=tokenizer.get_bos_token_id())
                    
                    # 创建批次
                    inputs_list = []
                    targets_list = []
                    
                    for tokens in tokens_list:
                        if len(tokens) > seq_len + 1:
                            tokens = tokens[:seq_len + 1]
                        elif len(tokens) < seq_len + 1:
                            # 填充到所需长度
                            tokens = tokens + [tokenizer.get_bos_token_id()] * (seq_len + 1 - len(tokens))
                        
                        inputs = torch.tensor(tokens[:-1], dtype=torch.int32)
                        targets = torch.tensor(tokens[1:], dtype=torch.int64)
                        
                        inputs_list.append(inputs)
                        targets_list.append(targets)
                    
                    if len(inputs_list) == 0:
                        continue
                        
                    # 堆叠并移动到NPU
                    inputs_batch = torch.stack(inputs_list).to(device)
                    targets_batch = torch.stack(targets_list).to(device)
                    
                    yield inputs_batch, targets_batch
                    
                except Exception as e:
                    print(f"数据处理错误: {e}")
                    continue
        
        data_loader = simple_data_loader()
        print("✅ 数据加载器创建成功")
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 创建优化器
    print("7. 创建优化器...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        print("✅ 优化器创建成功")
    except Exception as e:
        print(f"❌ 优化器创建失败: {e}")
        return
    
    # 8. 训练循环
    print("8. 开始训练...")
    try:
        model.train()
        autocast_ctx = torch.amp.autocast(device_type="npu", dtype=torch.bfloat16)
        
        for step, (x, y) in enumerate(data_loader):
            if step >= 10:  # 只训练10步作为测试
                break
                
            print(f"\n步骤 {step + 1}/10")
            print(f"  输入形状: {x.shape}, 设备: {x.device}")
            print(f"  目标形状: {y.shape}, 设备: {y.device}")
            
            # 前向传播
            torch_npu.npu.synchronize()
            t0 = time.time()
            
            with autocast_ctx:
                loss = model(x, y)
            
            print(f"  损失: {loss.item():.4f}")
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            torch_npu.npu.synchronize()
            t1 = time.time()
            
            dt = t1 - t0
            tokens_per_sec = (x.numel()) / dt
            memory_mb = torch_npu.npu.memory_allocated(0) / 1024**2
            
            print(f"  时间: {dt*1000:.1f}ms")
            print(f"  速度: {tokens_per_sec:,.0f} tokens/sec")
            print(f"  内存: {memory_mb:.1f}MB")
        
        print("\n🎉 训练测试完成！")
        print("NPU训练环境工作正常，可以开始正式训练了")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()