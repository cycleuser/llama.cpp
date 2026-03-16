# Hardware Acceleration Optimization Package

Cross-platform hardware acceleration optimizations for llama.cpp, supporting AMD GPUs, Apple Silicon, NVIDIA, Intel, and other mainstream hardware.

## Directory Structure

```
hardware-acceleration/
├── README.md              # Chinese Documentation
├── README_EN.md           # English Documentation (this file)
├── src/                   # Source code modifications
│   ├── ggml-cuda-common.cuh    # AMD GCN4 dp4a optimization
│   ├── ggml-metal-device.m     # Apple M-series detection
│   ├── ggml-opencl.cpp         # AMD OpenCL support
│   └── ggml-vulkan.cpp         # AMD Vulkan tuning
├── scripts/               # Utility scripts
│   ├── build.sh           # Build script
│   ├── test.sh            # Test script
│   └── ollama.sh          # Ollama model manager
└── docs/                  # Detailed documentation
```

## Quick Start

### 1. Apply Source Code Modifications

Copy files from `src/` to corresponding locations in llama.cpp:

```bash
# AMD GCN4 (RX580/RX590) dp4a optimization
cp src/ggml-cuda-common.cuh ../ggml/src/ggml-cuda/common.cuh

# Apple M-series detection
cp src/ggml-metal-device.m ../ggml/src/ggml-metal/ggml-metal-device.m

# AMD OpenCL support
cp src/ggml-opencl.cpp ../ggml/src/ggml-opencl/ggml-opencl.cpp

# AMD Vulkan tuning
cp src/ggml-vulkan.cpp ../ggml/src/ggml-vulkan/ggml-vulkan.cpp
```

### 2. Build

```bash
# Auto-detect hardware and build
./scripts/build.sh

# Manually specify backend
./scripts/build.sh metal    # Apple Metal
./scripts/build.sh hip      # AMD ROCm
./scripts/build.sh cuda     # NVIDIA
./scripts/build.sh vulkan   # Vulkan
```

### 3. Test

```bash
./scripts/test.sh
```

### 4. Use Ollama Models

```bash
# List models
./scripts/ollama.sh list

# Run model
./scripts/ollama.sh run library/gemma3:1b -p "Hello"

# Benchmark
./scripts/ollama.sh bench library/gemma3:1b
```

## Supported Hardware

| Platform | Backend | Notes |
|----------|---------|-------|
| Apple M1/M2/M3/M4 | Metal | Auto chip generation detection |
| AMD RX580/RX590 | HIP/Vulkan/OpenCL | GCN4 dp4a optimization |
| AMD RDNA Series | HIP/Vulkan | Native support |
| NVIDIA | CUDA | Most mature backend |
| Intel GPU | SYCL | oneAPI support |
| Generic | Vulkan/OpenCL | Cross-platform compatible |

## Source Code Modification Details

### ggml-cuda-common.cuh

Enables dp4a optimization for AMD GCN4 (gfx803: RX580/RX590):

```cpp
// Original
#elif defined(RDNA3) || defined(RDNA4)

// Modified
#elif defined(RDNA3) || defined(RDNA4) || defined(__gfx803__)
```

### ggml-metal-device.m

Adds Apple M-series chip detection:

```objc
// Detect chip generation
if (strstr(name, "M4")) { ... }
else if (strstr(name, "M3")) { ... }
// ...

// Output: Apple M4 series detected (10 GPU cores, 120 GB/s)
```

### ggml-opencl.cpp

Enables AMD GPU support:

```cpp
// Detect AMD GPU
if (vendor_str.find("AMD") != std::string::npos) {
    device->vendor = GGML_OPENCL_GPU_VENDOR_AMD;
    device->wavefront_size = get_amd_wavefront_size(device->name);
}
```

### ggml-vulkan.cpp

AMD GCN-specific tuning:

```cpp
// 64-wide wavefront optimization
if (is_amd_gcn(props)) {
    device->warptile_m = 64;
    device->warptile_n = 64;
}
```

## Environment Variables

```bash
# AMD HIP
export HSA_OVERRIDE_GFX_VERSION=8.0.3   # RX580/RX590

# AMD Vulkan
export GCN_SUBGROUP_SIZE=64              # GCN wavefront size
```

## Expected Performance

| Hardware | Backend | Expected Improvement |
|----------|---------|---------------------|
| RX580/RX590 | HIP | dp4a optimization, faster quantized models |
| Apple M4 | Metal | Auto detection, parameter tuning |
| AMD GCN | Vulkan | wavefront optimization |

## License

Same as main llama.cpp project, MIT License.