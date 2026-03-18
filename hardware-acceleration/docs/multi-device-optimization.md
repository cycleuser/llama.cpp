# 多设备协同推理优化方案

## 问题分析

### 当前架构限制

llama.cpp 支持三种并行模式：

| 模式 | 说明 | CPU+GPU 支持 |
|------|------|-------------|
| `LLAMA_SPLIT_MODE_NONE` | 单设备 | ❌ |
| `LLAMA_SPLIT_MODE_LAYER` | 层分割 | ✅ 支持 |
| `LLAMA_SPLIT_MODE_ROW` | 张量并行 | ❌ 仅 CUDA/SYCL |

### Apple Silicon 特殊情况

Apple M 系列芯片是**统一内存架构**：
- CPU 和 GPU 共享同一块物理内存
- 不需要 CPU↔GPU 数据传输
- Metal 后端自动处理内存共享

**这意味着：在 Apple Silicon 上，CPU 和 GPU "协同" 实际上没有意义，因为它们本来就是一体的！**

---

## 多设备协同方案

### 方案 1：层分割 (Layer Split) - 已支持

```
┌─────────────────────────────────────────────────────────┐
│                    大模型 (80层)                         │
├─────────────────────────────────────────────────────────┤
│  CPU: 层 0-19   │  GPU0: 层 20-49  │  GPU1: 层 50-79   │
│  (输入/输出)    │  (主要计算)      │  (主要计算)        │
└─────────────────────────────────────────────────────────┘
```

**使用方法：**
```bash
# 多 GPU 层分割
llama-cli -m model.gguf -ngl 80 --split-mode layer \
    --tensor-split 0.25,0.75  # GPU0: 25%, GPU1: 75%

# CPU + GPU 混合 (部分层在 CPU)
llama-cli -m model.gguf -ngl 60  # 60层在GPU, 20层在CPU
```

**优化建议：**
- 将计算密集的中间层放在 GPU
- 将 I/O 层（输入/输出）放在 CPU
- KV Cache 放在与对应层相同的设备

### 方案 2：流水线并行 (Pipeline Parallelism) - 部分支持

```
时间线:
T1: GPU0 计算层 0-19  ──────────────────►
T2:                      GPU1 计算层 20-39 ──────────────────►
T3:                                           GPU2 计算层 40-59 ──────►

流水线重叠:
T1: GPU0 [Batch1] ────────►
T2: GPU0 [Batch2] ────────► | GPU1 [Batch1] ────────►
T3: GPU0 [Batch3] ────────► | GPU1 [Batch2] ────────► | GPU2 [Batch1] ────►
```

**启用条件：**
1. 多个 GPU 设备
2. 所有层都卸载到 GPU (`-ngl > n_layer`)
3. 使用 Layer Split 模式
4. 所有设备支持 async 和 events

**当前限制：**
- CPU 后端不支持 pipeline parallelism
- 图重用与 pipeline parallelism 不兼容

### 方案 3：张量并行 (Tensor Parallelism) - 仅 CUDA

```
矩阵乘法 Y = X × W:
┌─────────────────────────────────────────┐
│  GPU0: Y0 = X × W0  (W 的上半部分)      │
│  GPU1: Y1 = X × W1  (W 的下半部分)      │
│  结果: Y = [Y0; Y1] (拼接)              │
└─────────────────────────────────────────┘
```

**仅支持：** NVIDIA CUDA, Intel SYCL

---

## 速度优化策略

### 策略 1：智能层分配

```python
# 伪代码：根据设备能力分配层
def assign_layers(model, devices):
    layers = model.n_layers
    
    # 计算每个设备的相对性能
    perf = [d.memory_bandwidth * d.compute_units for d in devices]
    total_perf = sum(perf)
    
    # 按性能比例分配层
    assignments = []
    for i, d in enumerate(devices):
        n = int(layers * perf[i] / total_perf)
        assignments.append((d, n))
    
    return assignments
```

### 策略 2：KV Cache 优化

```cpp
// KV Cache 应该放在与对应层相同的设备
// 避免跨设备内存访问

// 当前实现 (llama-kv-cache.cpp:121-126):
if (offload) {
    auto * dev = model.dev_layer(il);  // 与层同设备
    buft = ggml_backend_dev_buffer_type(dev);
}
```

### 策略 3：批量处理优化

```bash
# 增大 batch size 提高 GPU 利用率
llama-cli -m model.gguf -b 1024 -ngl 99

# 多序列并行处理
llama-cli -m model.gguf -n_parallel 4
```

### 策略 4：内存带宽优化

```
Token 生成速度 ≈ 内存带宽 / 模型大小

优化方向：
1. 使用更激进的量化 (Q3_K, Q2_K)
2. 使用 Flash Attention 减少 KV Cache 访问
3. 使用 Speculative Decoding (推测解码)
```

---

## 实际应用场景

### 场景 1：单 GPU + CPU (显存不足)

```bash
# 70B 模型，GPU 只有 16GB 显存
# 使用 CPU 处理部分层

llama-cli -m llama-70b-q4_k.gguf \
    -ngl 40 \              # 40层在 GPU
    -t 8 \                 # CPU 8 线程
    -b 512                 # batch size
```

**性能预期：**
- GPU 层：~80 t/s
- CPU 层：~5-10 t/s
- 整体：受 CPU 层限制

### 场景 2：多 GPU (同构)

```bash
# 2x RTX 4090 (各 24GB)
llama-cli -m llama-70b-q4_k.gguf \
    -ngl 80 \              # 全部层在 GPU
    --split-mode layer \   # 层分割
    --tensor-split 0.5,0.5 # 均匀分配
```

**性能预期：**
- 单 GPU：~50 t/s
- 双 GPU：~80-90 t/s (受同步开销影响)

### 场景 3：多 GPU (异构)

```bash
# RTX 4090 (24GB) + RTX 3080 (10GB)
llama-cli -m llama-70b-q4_k.gguf \
    -ngl 80 \
    --split-mode layer \
    --tensor-split 0.7,0.3  # 按显存比例分配
```

### 场景 4：Apple Silicon (统一内存)

```bash
# M4 Max (统一内存，无需分割)
llama-cli -m model.gguf -ngl 99 -t 8

# CPU 和 GPU 自动共享内存
# 无需手动配置
```

---

## 代码级优化建议

### 优化 1：动态层分配

```cpp
// 根据实时性能动态调整层分配
// 需要修改 llama-model.cpp

struct layer_assignment {
    int layer_idx;
    ggml_backend_dev_t device;
    float estimated_time;
};

std::vector<layer_assignment> optimize_layer_assignment(
    const llama_model & model,
    const std::vector<ggml_backend_dev_t> & devices) {
    
    // 1. 测量每个设备的基础性能
    // 2. 根据层大小估算计算时间
    // 3. 使用负载均衡算法分配
    // 4. 考虑数据传输开销
}
```

### 优化 2：异步流水线

```cpp
// 修改 ggml-backend.cpp 支持跨设备流水线

// 当前限制：CPU 不支持 async
// 解决方案：为 CPU 实现简化的 async 接口

struct ggml_backend_cpu_async {
    std::thread worker;
    std::queue<ggml_compute_params> work_queue;
    std::condition_variable cv;
};
```

### 优化 3：智能预取

```cpp
// 在计算当前层时，预取下一层的数据
// 对于非统一内存架构有效

void prefetch_next_layer(int current_layer) {
    int next_layer = current_layer + 1;
    auto * next_dev = get_device_for_layer(next_layer);
    if (next_dev != current_dev) {
        // 异步预取数据到目标设备
        async_copy(layer_data[next_layer], next_dev);
    }
}
```

---

## 总结

### Apple Silicon 用户

**好消息：** 统一内存架构已经是最优解！
- CPU 和 GPU 共享内存，零拷贝
- 无需手动配置多设备
- 只需 `-ngl 99` 全部卸载到 GPU

### 多 GPU 用户

1. **同构 GPU：** 使用 Layer Split + Pipeline Parallelism
2. **异构 GPU：** 使用 Layer Split + 按性能比例分配
3. **显存不足：** 部分层放 CPU，接受性能下降

### 未来优化方向

1. **Speculative Decoding：** 小模型猜测 + 大模型验证
2. **CPU async 支持：** 让 CPU 参与流水线并行
3. **动态负载均衡：** 实时调整层分配
4. **跨架构张量并行：** 支持 CPU+GPU 张量分割