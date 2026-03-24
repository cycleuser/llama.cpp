# AMD NPU Performance Guide

Optimize your AMD NPU inference for maximum performance.

## Model Selection

### Best Models for XDNA (Phoenix/Hawk Point)

Due to memory constraints (~2GB), small models work best:

| Model | Parameters | Memory | Performance |
|-------|------------|--------|-------------|
| Llama-3.2-1B-Instruct | 1B | ~1GB | Excellent |
| Qwen-2.5-1.5B | 1.5B | ~1.5GB | Excellent |
| Phi-3-mini-4k | 3.8B | ~2.5GB | Good (hybrid) |
| Gemma-2-2B | 2B | ~2GB | Good |

### Best Models for XDNA2 (Strix)

With 50+ TOPS and more memory:

| Model | Parameters | Memory | Performance |
|-------|------------|--------|-------------|
| Llama-3.2-3B-Instruct | 3B | ~3GB | Excellent |
| Qwen-2.5-3B | 3B | ~3GB | Excellent |
| Phi-3.5-mini | 3.8B | ~3GB | Excellent |
| Mistral-7B-v0.3 | 7B | ~5GB | Good (quantized) |

## Quantization

### INT8 Quantization (Recommended)

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize model for NPU
quantize_dynamic(
    "model.onnx",
    "model_int8.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['MatMul', 'Gemm']
)
```

### Expected Speedup

| Quantization | XDNA | XDNA2 |
|--------------|------|-------|
| FP32 | 1x | 1x |
| FP16 | 1.5x | 2x |
| INT8 | 2-3x | 3-4x |

## Configuration

### Environment Variables

```bash
# NPU device selection
export GGML_AMD_NPU_DEVICE=0

# Memory limit (bytes)
export GGML_AMD_NPU_MEMORY_LIMIT=2147483648  # 2GB

# Enable debug logging
export GGML_AMD_NPU_DEBUG=1

# Hybrid mode (Phoenix/Hawk Point)
export GGML_AMD_NPU_HYBRID=1
```

### Runtime Options

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()

# Optimize for NPU
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 1

# Enable memory pattern optimization
sess_options.enable_mem_pattern = True
sess_options.enable_mem_reuse = True
```

## Hybrid Execution (XDNA Gen 1)

For Phoenix/Hawk Point, use hybrid NPU+iGPU execution:

```python
# Layer split configuration
NPU_LAYERS = {
    'attention': 'npu',     # Attention on NPU
    'mlp': 'npu',           # MLP on NPU
    'embeddings': 'cpu',    # Embeddings on CPU
    'output': 'igpu'        # Output on integrated GPU
}

# Recommended layer distribution
npu_layers = 16   # First 16 layers on NPU
igpu_layers = 16  # Remaining on iGPU
```

### Performance Tuning

```bash
# For hybrid execution
cmake -B build \
    -DGGML_AMD_NPU=ON \
    -DGGML_HIP=ON \
    -DGGML_AMD_NPU_HYBRID=ON

# Run with hybrid mode
./llama-cli -m model.gguf -p "Hello" -ngl 32 --split-mode layer
```

## Batch Size Optimization

```python
# XDNA (Phoenix/Hawk Point)
batch_size = 1   # Optimal for small models
batch_size = 2   # May work with very small models

# XDNA2 (Strix)
batch_size = 1-4  # Depends on model size
batch_size = 8    # For very small models
```

## Memory Optimization

### Reduce Memory Usage

```python
# 1. Use INT8 quantization
# 2. Enable KV cache offloading
# 3. Use gradient checkpointing

# KV cache configuration
cache_config = {
    'kv_cache_size': 1024,  # Max sequence length
    'kv_cache_dtype': 'int8',  # Quantized cache
    'offload_kv_cache': True   # Offload to system memory
}
```

### Memory Estimation

```
Memory = Model_Weights + KV_Cache + Activations

For Llama-3.2-1B INT8:
- Model: ~1GB
- KV Cache (2048 ctx): ~256MB
- Activations: ~200MB
- Total: ~1.5GB (fits XDNA)

For Llama-3.2-3B INT8:
- Model: ~2.5GB
- KV Cache (2048 ctx): ~512MB
- Activations: ~400MB
- Total: ~3.5GB (needs XDNA2)
```

## Profiling

### Enable Profiling

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

session = ort.InferenceSession("model.onnx", sess_options)

# Run inference...

# Get profiling results
profile_file = session.end_profiling()
print(f"Profile saved to: {profile_file}")
```

### Analyze Results

```bash
# View profiling data
cat profile_*.json

# Common bottlenecks:
# - Memory transfers (high data_copy time)
# - Kernel launch overhead
# - CPU fallback for unsupported ops
```

## Benchmarking

### Standard Benchmark

```bash
# Run benchmark
python3 tools/benchmark_npu.py model.onnx -i 100 -d NPU

# Compare with CPU
python3 tools/benchmark_npu.py model.onnx --compare
```

### Expected Performance

**XDNA (Phoenix) - Llama-3.2-1B INT8:**
- Prefill: ~500 tokens/s
- Decode: ~30-40 tokens/s

**XDNA2 (Strix) - Llama-3.2-3B INT8:**
- Prefill: ~800 tokens/s
- Decode: ~50-60 tokens/s

## Troubleshooting

### Low Performance

1. Check NPU is actually being used:
```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
print(session.get_providers())  # Should show VitisAIExecutionProvider
```

2. Verify model is quantized:
```bash
# Check model size - INT8 should be ~1/4 of FP32
ls -lh model*.onnx
```

3. Check memory usage:
```bash
# Linux
watch -n 1 'cat /sys/class/drm/card*/device/mem_info_vram_total'

# Windows
# Use Task Manager > Performance > GPU
```

### Out of Memory

1. Reduce batch size
2. Use smaller context length
3. Enable memory offloading
4. Use more aggressive quantization

### Kernel Fallback

Check which operations fall back to CPU:
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced: Custom Kernels

For advanced users, you can compile custom kernels:

```bash
# Compile custom XCLBIN
cd hardware-acceleration/amd-npu
mkdir kernels && cd kernels

# Write kernel in C/C++ for AIE-ML
# Use Vitis HLS for compilation
v++ --mode aie --target hw \
    --platform xilinx_phx \
    --kernel my_kernel \
    -o my_kernel.xclbin
```

## References

- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [Vitis AI Optimization](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Optimizing-the-Model)
- [AMD AIE Programming](https://docs.xilinx.com/r/en-US/ug1079-ai-engine-environment)