# 硬件加速优化包

llama.cpp 跨平台硬件加速优化，支持 AMD GPU、Apple Silicon、NVIDIA、Intel 等主流硬件。

## 目录结构

```
hardware-acceleration/
├── README.md              # 中文说明
├── README_EN.md           # English Documentation
├── src/                   # 源代码修改
│   ├── ggml-cuda-common.cuh    # AMD GCN4 dp4a 优化
│   ├── ggml-metal-device.m     # Apple M 系列检测
│   ├── ggml-opencl.cpp         # AMD OpenCL 支持
│   └── ggml-vulkan.cpp         # AMD Vulkan 调优
├── scripts/               # 工具脚本
│   ├── build.sh           # 编译脚本
│   ├── test.sh            # 测试脚本
│   └── ollama.sh          # Ollama 模型管理
└── docs/                  # 详细文档
```

## 快速开始

### 1. 应用源代码修改

将 `src/` 目录下的文件复制到 llama.cpp 对应位置：

```bash
# AMD GCN4 (RX580/RX590) dp4a 优化
cp src/ggml-cuda-common.cuh ../ggml/src/ggml-cuda/common.cuh

# Apple M 系列检测
cp src/ggml-metal-device.m ../ggml/src/ggml-metal/ggml-metal-device.m

# AMD OpenCL 支持
cp src/ggml-opencl.cpp ../ggml/src/ggml-opencl/ggml-opencl.cpp

# AMD Vulkan 调优
cp src/ggml-vulkan.cpp ../ggml/src/ggml-vulkan/ggml-vulkan.cpp
```

### 2. 编译

```bash
# 自动检测硬件并编译
./scripts/build.sh

# 手动指定后端
./scripts/build.sh metal    # Apple Metal
./scripts/build.sh hip      # AMD ROCm
./scripts/build.sh cuda     # NVIDIA
./scripts/build.sh vulkan   # Vulkan
```

### 3. 测试

```bash
./scripts/test.sh
```

### 4. 使用 Ollama 模型

```bash
# 列出模型
./scripts/ollama.sh list

# 运行模型
./scripts/ollama.sh run library/gemma3:1b -p "你好"

# 性能测试
./scripts/ollama.sh bench library/gemma3:1b
```

## 支持的硬件

| 平台 | 后端 | 说明 |
|------|------|------|
| Apple M1/M2/M3/M4 | Metal | 自动检测芯片代数 |
| AMD RX580/RX590 | HIP/Vulkan/OpenCL | GCN4 dp4a 优化 |
| AMD RDNA 系列 | HIP/Vulkan | 原生支持 |
| NVIDIA | CUDA | 最成熟的后端 |
| Intel GPU | SYCL | oneAPI 支持 |
| 通用 | Vulkan/OpenCL | 跨平台兼容 |

## 源代码修改说明

### ggml-cuda-common.cuh

为 AMD GCN4 (gfx803: RX580/RX590) 启用 dp4a 优化：

```cpp
// 原代码
#elif defined(RDNA3) || defined(RDNA4)

// 修改后
#elif defined(RDNA3) || defined(RDNA4) || defined(__gfx803__)
```

### ggml-metal-device.m

添加 Apple M 系列芯片检测：

```objc
// 检测芯片代数
if (strstr(name, "M4")) { ... }
else if (strstr(name, "M3")) { ... }
// ...

// 输出: Apple M4 series detected (10 GPU cores, 120 GB/s)
```

### ggml-opencl.cpp

启用 AMD GPU 支持：

```cpp
// 检测 AMD GPU
if (vendor_str.find("AMD") != std::string::npos) {
    device->vendor = GGML_OPENCL_GPU_VENDOR_AMD;
    device->wavefront_size = get_amd_wavefront_size(device->name);
}
```

### ggml-vulkan.cpp

AMD GCN 特定调优：

```cpp
// 64 宽度 wavefront 优化
if (is_amd_gcn(props)) {
    device->warptile_m = 64;
    device->warptile_n = 64;
}
```

## 环境变量

```bash
# AMD HIP
export HSA_OVERRIDE_GFX_VERSION=8.0.3   # RX580/RX590

# AMD Vulkan
export GCN_SUBGROUP_SIZE=64              # GCN wavefront 大小
```

## 性能预期

| 硬件 | 后端 | 预期提升 |
|------|------|----------|
| RX580/RX590 | HIP | dp4a 优化，量化模型提速 |
| Apple M4 | Metal | 自动检测，参数调优 |
| AMD GCN | Vulkan | wavefront 优化 |

## 许可证

与 llama.cpp 主项目相同，MIT 许可证。