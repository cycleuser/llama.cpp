# Hardware Acceleration Package

Cross-platform hardware acceleration optimizations for llama.cpp, supporting AMD GPU, Apple Silicon, NVIDIA, Intel, and other mainstream hardware.

## New Features

### Multi-Device Parallel Inference

Run multiple models simultaneously on different devices without interference:

```bash
# Run two models on CPU and GPU in parallel
./multi-device-runner -m model1.gguf -d cpu -m model2.gguf -d gpu0

# List all available devices
./multi-device-runner -l
```

### Automatic Hardware Detection

```bash
./scripts/detect-devices.sh
```

## Directory Structure

```
hardware-acceleration/
├── README.md                    # Chinese Documentation
├── README_EN.md                 # English Documentation
├── src/                         # Source Code Modifications
│   ├── ggml-cuda-common.cuh     # AMD GCN4 dp4a optimization
│   ├── ggml-metal-device.m      # Apple M-series detection
│   ├── ggml-opencl.cpp          # AMD OpenCL support
│   ├── ggml-vulkan.cpp          # AMD Vulkan tuning
│   └── cpu-optimizations.cpp    # CPU performance optimizations
├── tools/                       # Tools
│   └── multi-device-runner.cpp  # Multi-device parallel inference tool
└── scripts/                     # Scripts
    ├── build.sh                 # Build script
    ├── test.sh                  # Test script
    ├── ollama.sh                # Ollama model manager
    ├── detect-devices.sh        # Hardware detection
    └── benchmark.sh             # Performance benchmark
```

## Quick Start

### 1. Detect Hardware

```bash
./scripts/detect-devices.sh
```

### 2. Build

```bash
# Apple Silicon (M1/M2/M3/M4)
cmake -B build && cmake --build build -j

# NVIDIA GPU
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# AMD GPU (ROCm/HIP)
cmake -B build -DGGML_HIP=ON && cmake --build build -j
# For RX580/RX590, also set:
export HSA_OVERRIDE_GFX_VERSION=8.0.3

# Vulkan (cross-platform)
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
```

### 3. Multi-Device Parallel Inference

```bash
# Compile the multi-device tool
cd tools
g++ -O3 -std=c++17 -I../../ -I../../src -I../../ggml/include \
    multi-device-runner.cpp -o multi-device-runner \
    -L../../build/src -lllama -L../../build/ggml/src -lggml \
    -L../../build/ggml/src/ggml-cpu -lggml-cpu \
    -lpthread -ldl

# Usage examples
./multi-device-runner -l                              # List devices
./multi-device-runner -m model.gguf -d cpu            # CPU only
./multi-device-runner -m a.gguf -d cpu -m b.gguf -d gpu0  # CPU + GPU parallel
```

## Supported Hardware

| Platform | Backend | Notes |
|----------|---------|-------|
| Apple M1/M2/M3/M4 | Metal | Auto-detect chip generation, extreme optimization |
| AMD RX580/RX590 | HIP/Vulkan/OpenCL | GCN4 dp4a optimization |
| AMD RDNA Series | HIP/Vulkan | Native support |
| NVIDIA | CUDA | Most mature backend |
| Intel GPU | SYCL | oneAPI support |
| General | Vulkan/OpenCL | Cross-platform compatibility |

## Performance Optimization Tips

### General Optimizations

1. **Batch Size**: Increase `-b` parameter for better throughput
   ```bash
   llama-cli -m model.gguf -b 1024 -n 128
   ```

2. **Thread Count**: For CPU inference, threads should match physical cores
   ```bash
   llama-cli -m model.gguf -t 8  # 8-core CPU
   ```

3. **GPU Layers**: Offload all layers to GPU for best performance
   ```bash
   llama-cli -m model.gguf -ngl 99
   ```

### Apple Silicon Optimization

1. Metal backend is automatically enabled
2. Use `-ngl 99` to offload all layers to GPU
3. Unified memory architecture - no memory copy overhead

### AMD GPU Optimization

1. **RX580/RX590 (gfx803)**:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=8.0.3
   export GPU_MAX_HW_QUEUES=2
   ```

2. **RDNA Series**: Use directly, no extra settings needed

3. **Vulkan Backend**: May be more stable for some AMD GPUs
   ```bash
   cmake -B build -DGGML_VULKAN=ON
   ```

### NVIDIA GPU Optimization

1. Use CUDA 12.x for best performance
2. Flash Attention is automatically enabled
3. Multi-GPU with `--split-mode layer`

## Multi-Device Parallel Architecture

### Architecture Overview

llama.cpp supports three parallel modes:

1. **Single Model, Multiple GPUs**: Distribute model layers across GPUs
   ```bash
   llama-cli -m model.gguf -ngl 99 --split-mode layer
   ```

2. **Multiple Models, Multiple Devices**: Different models on different devices
   ```bash
   ./multi-device-runner -m a.gguf -d cpu -m b.gguf -d gpu0
   ```

3. **Multiple Sequences, Single Context**: One context handling multiple conversations
   ```bash
   llama-cli -m model.gguf -n_parallel 4
   ```

### Code Example

```cpp
// Run two models on CPU and GPU in parallel
std::vector<ggml_backend_dev_t> devices = detect_devices();

// Model 1: CPU
llama_model_params mparams1 = llama_model_default_params();
mparams1.split_mode = LLAMA_SPLIT_MODE_NONE;
mparams1.devices = &cpu_device;
llama_model* model1 = llama_model_load_from_file(path1, mparams1);

// Model 2: GPU
llama_model_params mparams2 = llama_model_default_params();
mparams2.split_mode = LLAMA_SPLIT_MODE_NONE;
mparams2.devices = &gpu_device;
llama_model* model2 = llama_model_load_from_file(path2, mparams2);

// Parallel inference
std::thread t1([&]() { run_inference(model1); });
std::thread t2([&]() { run_inference(model2); });
t1.join();
t2.join();
```

## Source Code Modifications

### ggml-cuda-common.cuh

Enable dp4a optimization for AMD GCN4 (gfx803: RX580/RX590):

```cpp
// Original
#elif defined(RDNA3) || defined(RDNA4)

// Modified
#elif defined(RDNA3) || defined(RDNA4) || defined(__gfx803__)
```

### ggml-opencl.cpp

Add AMD/NVIDIA GPU detection:

```cpp
enum GPU_FAMILY {
    ADRENO,
    AMD,      // Added
    INTEL,
    NVIDIA,   // Added
    UNKNOWN,
};
```

### cpu-optimizations.cpp

CPU performance optimization patches:
- SIMD-optimized RMS Norm
- Improved batch memory allocation
- Thread pool warmup

## Environment Variables

```bash
# AMD HIP
export HSA_OVERRIDE_GFX_VERSION=8.0.3   # RX580/RX590

# AMD Vulkan
export GCN_SUBGROUP_SIZE=64              # GCN wavefront size
```

## FAQ

### Q: Can multiple models share GPU memory?

A: Yes, but watch out for VRAM limits. Use `--mlock` to lock model memory.

### Q: How to choose the best backend?

A: 
- Apple Silicon: Metal (automatic)
- NVIDIA: CUDA
- AMD: HIP (recommended) or Vulkan
- Intel: SYCL
- Others: Vulkan

### Q: RX580/RX590 inference is slow, what to do?

A: 
1. Make sure `HSA_OVERRIDE_GFX_VERSION=8.0.3` is set
2. Vulkan backend may be more stable
3. Reducing GPU layers `-ngl 32` may be faster

## Changelog

### 2026-03-18
- Added multi-device parallel inference tool
- Added hardware detection script
- Added performance benchmark script
- Added CPU performance optimization patches
- Updated AMD GPU support (OpenCL backend)

## License

Same as main llama.cpp project, MIT License.