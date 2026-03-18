# 动态多设备调度方案

## 当前状态：静态分配

```cpp
// src/llama-model.cpp:2636-2647
// 层分配在模型加载时确定，运行时不变

const int i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        return {cpu_dev, &pimpl->cpu_buft_list};  // CPU
    }
    auto * dev = devices.at(layer_gpu);  // GPU
    return {dev, &pimpl->gpu_buft_list.at(dev)};
};
```

**问题：** 一旦分配，无法运行时调整。

---

## 动态调度方案

### 方案 1：基于负载的动态迁移

```
┌─────────────────────────────────────────────────────────────┐
│                    动态调度监控器                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│   │ 监控    │───▶│ 决策    │───▶│ 迁移    │               │
│   │ GPU负载 │    │ 算法    │    │ 层数据  │               │
│   └─────────┘    └─────────┘    └─────────┘               │
│                                                             │
│   监控指标:                                                  │
│   - GPU 利用率                                               │
│   - 内存带宽使用                                              │
│   - 队列深度                                                  │
│   - 温度/功耗                                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 方案 2：流水线动态重平衡

```
初始状态:
  GPU0: 层 0-19  [████████████] 负载 90%
  GPU1: 层 20-39 [████░░░░░░░░] 负载 40%

检测到不平衡 → 动态重分配:
  GPU0: 层 0-14  [████████░░░░] 负载 60%
  GPU1: 层 15-39 [████████████] 负载 70%
```

### 实现代码框架

```cpp
// 动态调度器伪代码
class DynamicLayerScheduler {
    struct DeviceMetrics {
        float utilization;      // GPU 利用率
        float memory_used;      // 内存使用
        float temperature;      // 温度
        double avg_latency;     // 平均延迟
    };

    std::vector<DeviceMetrics> device_metrics;
    std::mutex metrics_mutex;
    std::atomic<bool> running{true};
    std::thread monitor_thread;

    void monitor_loop() {
        while (running) {
            // 1. 收集所有设备指标
            for (auto& dev : devices) {
                auto metrics = collect_metrics(dev);
                update_device_metrics(dev, metrics);
            }

            // 2. 分析负载均衡情况
            auto imbalance = calculate_imbalance();

            // 3. 如果不平衡超过阈值，触发迁移
            if (imbalance > threshold) {
                rebalance_layers();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void rebalance_layers() {
        // 计算新的层分配
        auto new_assignment = compute_optimal_assignment();

        // 异步迁移层数据
        for (auto& migration : new_assignment.migrations) {
            async_migrate_layer(
                migration.layer_id,
                migration.from_device,
                migration.to_device
            );
        }

        // 等待迁移完成
        wait_for_migrations();

        // 更新调度器状态
        update_layer_assignment(new_assignment);
    }
};
```

### 挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| 数据迁移开销 | 预测性迁移，空闲时迁移 |
| 状态同步 | 双缓冲，原子切换 |
| 内存碎片 | 内存池管理 |
| 中断推理 | 检查点机制 |

### 性能影响评估

```
静态分配 vs 动态调度:

场景 1: 均衡负载
  静态: 100 t/s
  动态: 98 t/s (监控开销)

场景 2: 不均衡负载 (GPU0 90%, GPU1 30%)
  静态: 60 t/s (受慢设备限制)
  动态: 85 t/s (重平衡后)

场景 3: 动态变化负载
  静态: 波动大
  动态: 稳定在 90%+ 效率
```

---

## Apple Silicon 特殊情况

**统一内存架构下，动态调度意义有限：**

```
Apple M4:
- CPU 和 GPU 共享同一块物理内存
- 无需数据拷贝
- 任何"迁移"只是改变计算设备，数据不动

动态调度的价值：
- 在 CPU 和 GPU 之间动态切换计算单元
- 当 GPU 负载高时，部分层切换到 CPU
- 但 CPU 计算效率远低于 GPU，通常不划算
```

---

## 实现建议

### 短期可行方案

1. **批处理时动态调整**
   - 每个 batch 之间重新评估分配
   - 避免推理中途迁移

2. **基于队列深度调度**
   - 监控每个设备的请求队列
   - 新请求发送到最空闲设备

3. **温度感知调度**
   - 高温时减少该设备负载
   - 防止热节流

### 中期方案

1. **预测性调度**
   - 使用机器学习预测负载
   - 提前迁移数据

2. **自适应流水线**
   - 动态调整流水线深度
   - 根据实时性能调整

### 长期方案

1. **完全动态系统**
   - 实时监控、决策、迁移
   - 自动负载均衡

---

## 代码实现路径

```
步骤 1: 添加设备监控 API
├── ggml_backend_get_device_metrics(dev, &metrics)
└── 暴露利用率、内存、温度

步骤 2: 实现迁移机制
├── ggml_backend_migrate_tensor(tensor, from_dev, to_dev)
└── 异步数据传输

步骤 3: 添加调度器
├── llama_dynamic_scheduler
└── 后台线程监控和决策

步骤 4: 集成到推理循环
├── 检查点机制
└── 安全迁移点
```