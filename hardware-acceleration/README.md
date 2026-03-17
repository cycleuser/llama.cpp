# 硬件加速优化包

llama.cpp 跨平台硬件加速优化，支持 AMD GPU、Apple Silicon、NVIDIA、Intel 等主流硬件。

## 新功能

### 多设备并行推理

现在可以在不同设备上同时运行多个模型，互不干扰：

```bash
# 在 CPU 和 GPU 上同时运行两个模型
./multi-device-runner -m model1.gguf -d cpu -m model2.gguf -d gpu0

# 列出所有可用设备
./multi-device-runner -l
```

### 自动硬件检测

```bash
./scripts/detect-devices.sh
```

## 目录结构

```
hardware-acceleration/
├── README.md                    # 中文说明
├── README_EN.md                 # English Documentation
├── src/                         # 源代码修改
│   ├── ggml-cuda-common.cuh     # AMD GCN4 dp4a 优化
│   ├── ggml-metal-device.m      # Apple M 系列检测
│   ├── ggml-opencl.cpp          # AMD OpenCL 支持
│   ├── ggml-vulkan.cpp          # AMD Vulkan 调优
│   └── cpu-optimizations.cpp    # CPU 性能优化
├── tools/                       # 工具程序
│   └── multi-device-runner.cpp  # 多设备并行推理工具
└── scripts/                     # 工具脚本
    ├── build.sh                 # 编译脚本
    ├── test.sh                  # 测试脚本
    ├── ollama.sh                # Ollama 模型管理
    ├── detect-devices.sh        # 硬件检测
    └── benchmark.sh             # 性能基准测试
```

## 快速开始

### 1. 检测硬件

```bash
./scripts/detect-devices.sh
```

### 2. 编译

```bash
# Apple Silicon (M1/M2/M3/M4)
cmake -B build && cmake --build build -j

# NVIDIA GPU
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# AMD GPU (ROCm/HIP)
cmake -B build -DGGML_HIP=ON && cmake --build build -j
# RX580/RX590 需要额外设置:
export HSA_OVERRIDE_GFX_VERSION=8.0.3

# Vulkan (跨平台)
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
```

### 3. 多设备并行推理

```bash
# 编译多设备工具
cd tools
g++ -O3 -std=c++17 -I../../ -I../../src -I../../ggml/include \
    multi-device-runner.cpp -o multi-device-runner \
    -L../../build/src -lllama -L../../build/ggml/src -lggml \
    -L../../build/ggml/src/ggml-cpu -lggml-cpu \
    -lpthread -ldl

# 使用示例
./multi-device-runner -l                              # 列出设备
./multi-device-runner -m model.gguf -d cpu            # 仅 CPU
./multi-device-runner -m a.gguf -d cpu -m b.gguf -d gpu0  # CPU + GPU 并行
```

## 支持的硬件

| 平台 | 后端 | 说明 |
|------|------|------|
| Apple M1/M2/M3/M4 | Metal | 自动检测芯片代数，极端优化 |
| AMD RX580/RX590 | HIP/Vulkan/OpenCL | GCN4 dp4a 优化 |
| AMD RDNA 系列 | HIP/Vulkan | 原生支持 |
| NVIDIA | CUDA | 最成熟的后端 |
| Intel GPU | SYCL | oneAPI 支持 |
| 通用 | Vulkan/OpenCL | 跨平台兼容 |

## 性能优化建议

### 通用优化

1. **批量大小**: 增大 `-b` 参数可以提高吞吐量
   ```bash
   llama-cli -m model.gguf -b 1024 -n 128
   ```

2. **线程数**: CPU 推理时，线程数应接近物理核心数
   ```bash
   llama-cli -m model.gguf -t 8  # 8 核 CPU
   ```

3. **GPU 层数**: 全部卸载到 GPU 获得最佳性能
   ```bash
   llama-cli -m model.gguf -ngl 99
   ```

### Apple Silicon 优化

1. Metal 后端自动启用，无需额外配置
2. 使用 `-ngl 99` 将所有层卸载到 GPU
3. 统一内存架构，无需担心内存拷贝

### AMD GPU 优化

1. **RX580/RX590 (gfx803)**:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=8.0.3
   export GPU_MAX_HW_QUEUES=2
   ```

2. **RDNA 系列**: 直接使用，无需额外设置

3. **Vulkan 后端**: 对于某些 AMD GPU 可能比 HIP 更稳定
   ```bash
   cmake -B build -DGGML_VULKAN=ON
   ```

### NVIDIA GPU 优化

1. 使用 CUDA 12.x 获得最佳性能
2. Flash Attention 自动启用
3. 多 GPU 可以使用 `--split-mode layer`

## 多设备并行架构

### 架构说明

llama.cpp 支持三种并行模式：

1. **单模型多GPU**: 将模型层分布到多个GPU
   ```bash
   llama-cli -m model.gguf -ngl 99 --split-mode layer
   ```

2. **多模型多设备**: 不同模型运行在不同设备
   ```bash
   ./multi-device-runner -m a.gguf -d cpu -m b.gguf -d gpu0
   ```

3. **多序列单上下文**: 一个上下文处理多个对话
   ```bash
   llama-cli -m model.gguf -n_parallel 4
   ```

### 代码示例

```cpp
// 运行两个模型，分别在 CPU 和 GPU
std::vector<ggml_backend_dev_t> devices = detect_devices();

// 模型1: CPU
llama_model_params mparams1 = llama_model_default_params();
mparams1.split_mode = LLAMA_SPLIT_MODE_NONE;
mparams1.devices = &cpu_device;
llama_model* model1 = llama_model_load_from_file(path1, mparams1);

// 模型2: GPU
llama_model_params mparams2 = llama_model_default_params();
mparams2.split_mode = LLAMA_SPLIT_MODE_NONE;
mparams2.devices = &gpu_device;
llama_model* model2 = llama_model_load_from_file(path2, mparams2);

// 并行推理
std::thread t1([&]() { run_inference(model1); });
std::thread t2([&]() { run_inference(model2); });
t1.join();
t2.join();
```

## 源代码修改说明

### ggml-cuda-common.cuh

为 AMD GCN4 (gfx803: RX580/RX590) 启用 dp4a 优化：

```cpp
// 原代码
#elif defined(RDNA3) || defined(RDNA4)

// 修改后
#elif defined(RDNA3) || defined(RDNA4) || defined(__gfx803__)
```

### ggml-opencl.cpp

添加 AMD/NVIDIA GPU 检测：

```cpp
enum GPU_FAMILY {
    ADRENO,
    AMD,      // 新增
    INTEL,
    NVIDIA,   // 新增
    UNKNOWN,
};
```

### cpu-optimizations.cpp

CPU 性能优化补丁：
- SIMD 优化的 RMS Norm
- 改进的批处理内存分配
- 线程池预热

## 常见问题

### Q: 多个模型能否共享 GPU 内存？

A: 可以，但需要注意显存限制。使用 `--mlock` 锁定模型内存。

### Q: 如何选择最佳后端？

A: 
- Apple Silicon: Metal (自动)
- NVIDIA: CUDA
- AMD: HIP (推荐) 或 Vulkan
- Intel: SYCL
- 其他: Vulkan

### Q: RX580/RX590 推理速度慢怎么办？

A: 
1. 确保设置了 `HSA_OVERRIDE_GFX_VERSION=8.0.3`
2. 使用 Vulkan 后端可能更稳定
3. 减少 GPU 层数 `-ngl 32` 可能更快

## 更新日志

### 2026-03-18
- 添加多设备并行推理工具
- 添加硬件检测脚本
- 添加性能基准测试脚本
- 添加 CPU 性能优化补丁
- 更新 AMD GPU 支持 (OpenCL 后端)