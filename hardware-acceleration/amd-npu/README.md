# AMD NPU (XDNA/Ryzen AI) Support for llama.cpp

Comprehensive support for AMD Neural Processing Unit acceleration on Ryzen AI processors.

## Hardware Compatibility

### XDNA Architecture (Gen 1)
| Processor | Codename | NPU TOPS | LLM Support |
|-----------|----------|----------|-------------|
| Ryzen 7 7840HS | Phoenix | 10 | Hybrid* |
| Ryzen 7 8840HS | Hawk Point | 16 | Hybrid* |
| Ryzen 5 7640HS | Phoenix | 10 | Hybrid* |
| Ryzen 9 7940HS | Phoenix | 10 | Hybrid* |
| Ryzen 9 8945HS | Hawk Point | 16 | Hybrid* |

*Hybrid: NPU for pre-processing/attention, CPU/iGPU for token generation

### XDNA2 Architecture (Gen 2)
| Processor | Codename | NPU TOPS | LLM Support |
|-----------|----------|----------|-------------|
| Ryzen AI 9 HX 370 | Strix | 50 | Full NPU |
| Ryzen AI 9 365 | Strix | 50 | Full NPU |
| Ryzen AI Max+ 395 | Strix Halo | 75 | Full NPU |
| Ryzen AI 7 PRO 360 | Krackan | 50 | Full NPU |

## Quick Start

### Windows
```powershell
# 1. Install NPU driver
# Download from AMD website or Windows Update

# 2. Install Ryzen AI Software
# Download: https://www.amd.com/en/developer/resources/ryzen-ai.html

# 3. Verify installation
python scripts\verify_npu.py
```

### Linux (Ubuntu 24.04+)
```bash
# 1. Install kernel 6.14+ with amdxdna driver
sudo apt install linux-generic-hwe-24.04

# 2. Install XRT
./scripts/install-xrt.sh

# 3. Verify
python3 scripts/verify_npu.py
```

## Backend Options

### Option 1: ONNX Runtime + Vitis AI EP (Recommended)
- Full NPU utilization
- Automatic model optimization
- Hybrid NPU+iGPU support

### Option 2: Direct XRT Integration
- Lower latency
- More control
- Requires model compilation

### Option 3: IREE-AMD-AIE
- Open source
- MLIR-based
- Best for research

## Directory Structure

```
amd-npu/
├── src/
│   ├── ggml-amdxdna.cpp      # Backend implementation
│   ├── ggml-amdxdna.h        # Header file
│   ├── xrt_wrapper.cpp       # XRT API wrapper
│   └── vitis_ai_ep.cpp       # Vitis AI EP integration
├── tools/
│   ├── gguf_to_onnx.py       # Model converter
│   ├── compile_model.py      # NPU model compiler
│   └── benchmark_npu.py      # Performance testing
├── scripts/
│   ├── install-xrt.sh        # XRT installation
│   ├── detect-npu.sh         # Hardware detection
│   └── verify_npu.py         # Installation check
├── docs/
│   ├── installation.md       # Detailed installation guide
│   ├── architecture.md       # XDNA architecture overview
│   └── performance.md        # Optimization guide
└── models/
    └── optimized/            # Pre-compiled models
```

## Performance Expectations

| Model | Phoenix (NPU+iGPU) | Hawk Point (NPU+iGPU) | Strix (NPU Only) |
|-------|--------------------|-----------------------|------------------|
| Llama-3.2-1B | ~15 t/s | ~20 t/s | ~35 t/s |
| Llama-3.2-3B | ~8 t/s | ~12 t/s | ~25 t/s |
| Qwen-2.5-1.5B | ~18 t/s | ~24 t/s | ~40 t/s |
| Phi-3-mini | ~12 t/s | ~16 t/s | ~30 t/s |

## Known Limitations

1. **Phoenix/Hawk Point**: LLM requires hybrid execution (NPU + iGPU)
2. **Memory**: NPU has limited memory (~2GB), large models need memory management
3. **Quantization**: INT8 models work best on NPU
4. **Drivers**: Windows support is more mature than Linux

## References

- [Ryzen AI Documentation](https://ryzenai.docs.amd.com)
- [XDNA Driver](https://github.com/amd/xdna-driver)
- [AMD whisper.cpp](https://github.com/amd/whisper.cpp)
- [RyzenAI-SW Examples](https://github.com/amd/RyzenAI-SW)