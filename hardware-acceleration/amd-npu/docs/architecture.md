# AMD XDNA Architecture Overview

Deep dive into AMD's XDNA NPU architecture and its capabilities.

## Architecture Comparison

| Feature | XDNA (Gen 1) | XDNA2 (Gen 2) |
|---------|-------------|---------------|
| Codename | Phoenix/Hawk Point | Strix/Strix Halo |
| AIE-ML Tiles | 4×4 (16 tiles) | 8×4 (32 tiles) |
| TOPS (INT8) | 10-16 | 50-75 |
| Memory | ~2GB shared | ~4GB shared |
| BF16 Support | No | Yes |
| LLM Support | Hybrid only | Full NPU |

## AIE-ML Architecture

### Core Components

```
┌─────────────────────────────────────────────┐
│                  XDNA NPU                    │
│  ┌─────────┬─────────┬─────────┬─────────┐  │
│  │ AIE-ML  │ AIE-ML  │ AIE-ML  │ AIE-ML  │  │
│  │  Tile   │  Tile   │  Tile   │  Tile   │  │
│  │  [0,0]  │  [0,1]  │  [0,2]  │  [0,3]  │  │
│  ├─────────┼─────────┼─────────┼─────────┤  │
│  │ AIE-ML  │ AIE-ML  │ AIE-ML  │ AIE-ML  │  │
│  │  Tile   │  Tile   │  Tile   │  Tile   │  │
│  │  [1,0]  │  [1,1]  │  [1,2]  │  [1,3]  │  │
│  ├─────────┼─────────┼─────────┼─────────┤  │
│  │ AIE-ML  │ AIE-ML  │ AIE-ML  │ AIE-ML  │  │
│  │  Tile   │  Tile   │  Tile   │  Tile   │  │
│  │  [2,0]  │  [2,1]  │  [2,2]  │  [2,3]  │  │
│  ├─────────┼─────────┼─────────┼─────────┤  │
│  │ AIE-ML  │ AIE-ML  │ AIE-ML  │ AIE-ML  │  │
│  │  Tile   │  Tile   │  Tile   │  Tile   │  │
│  │  [3,0]  │  [3,1]  │  [3,2]  │  [3,3]  │  │
│  └─────────┴─────────┴─────────┴─────────┘  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         Data Memory (32KB/tile)     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         Program Memory (16KB/tile)  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### AIE-ML Tile Capabilities

Each AIE-ML tile can perform:
- **MAC Operations**: 256 INT8 MACs per cycle
- **Vector Operations**: SIMD operations for ML workloads
- **Data Movement**: DMA for efficient memory transfers
- **Local Memory**: 32KB data + 16KB program memory

### Data Flow

```
DDR Memory → Shim DMA → AIE-ML Tiles → Shim DMA → DDR Memory
                   ↓
            Inter-tile streams
            (neighbor communication)
```

## Performance Characteristics

### Compute Throughput

| Operation | XDNA (TOPS) | XDNA2 (TOPS) |
|-----------|-------------|--------------|
| INT8 MAC | 10-16 | 50-75 |
| BF16 MAC | N/A | 25-37.5 |
| INT16 MAC | 5-8 | 25-37.5 |

### Memory Bandwidth

```
Shared Memory Architecture:
- NPU shares system memory with CPU
- No dedicated VRAM
- Typical bandwidth: 50-100 GB/s (LPDDR5X)

Memory Hierarchy:
1. System RAM (DDR5/LPDDR5X) - Large, slower
2. NPU Local Memory (~2-4MB) - Small, fast
3. Tile Data Memory (32KB × 16 tiles) - Fastest
```

### Latency Characteristics

| Operation | Latency |
|-----------|---------|
| Tile-to-Tile | ~10 cycles |
| Local Memory | ~1 cycle |
| System Memory | ~100-200 cycles |

## Programming Model

### Software Stack

```
┌─────────────────────────────────────────────┐
│           Application (llama.cpp)           │
├─────────────────────────────────────────────┤
│              GGML Backend                   │
├─────────────────────────────────────────────┤
│           ONNX Runtime GenAI                │
├─────────────────────────────────────────────┤
│         Vitis AI Execution Provider         │
├─────────────────────────────────────────────┤
│                  XRT                        │
├─────────────────────────────────────────────┤
│              amdxdna driver                 │
├─────────────────────────────────────────────┤
│              NPU Hardware                   │
└─────────────────────────────────────────────┘
```

### Supported Operations

**Fully Supported (Hardware Accelerated)**
- Matrix Multiplication (MatMul)
- Convolution (Conv1D, Conv2D)
- Element-wise operations (Add, Mul, Sub)
- Activation functions (ReLU, GELU, SiLU, Sigmoid)
- Normalization (LayerNorm, RMSNorm)
- Softmax
- Pooling (MaxPool, AvgPool)

**Partially Supported (May Fall Back to CPU)**
- Attention mechanisms (requires optimization)
- Dynamic shapes
- Custom operations

**Not Supported**
- High-precision float (F64)
- Complex numbers
- Sparse operations

## Model Optimization

### Quantization

For best NPU performance, use INT8 quantization:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_int8.onnx",
    weight_type=QuantType.QInt8
)
```

### Graph Optimization

```python
from onnx import optimizer

optimized_model = optimizer.optimize(
    model,
    passes=[
        'eliminate_identity',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
    ]
)
```

### Memory Planning

1. **Tile Memory Usage**: Keep tile data < 32KB
2. **Batch Size**: Start with 1, increase if memory allows
3. **Sequence Length**: Consider chunking for long sequences

## Hybrid Execution

### Phoenix/Hawk Point Limitations

XDNA Gen 1 has limited memory and compute for full LLM execution:

```
┌──────────────────────────────────────────────┐
│            Hybrid Execution Flow             │
│                                              │
│  ┌─────────────┐      ┌─────────────────┐   │
│  │  Prefill    │ ───→ │      NPU        │   │
│  │  (Prompt)   │      │  (Attention)    │   │
│  └─────────────┘      └─────────────────┘   │
│                              │               │
│                              ↓               │
│  ┌─────────────┐      ┌─────────────────┐   │
│  │  Decode     │ ───→ │  iGPU / CPU     │   │
│  │  (Tokens)   │      │  (Generation)   │   │
│  └─────────────┘      └─────────────────┘   │
│                                              │
└──────────────────────────────────────────────┘
```

### Recommended Split

| Layer Type | XDNA Gen 1 | XDNA2 |
|------------|------------|-------|
| Embedding | CPU | NPU |
| Attention | NPU | NPU |
| MLP | NPU | NPU |
| LayerNorm | NPU | NPU |
| Output | iGPU | NPU |

## Performance Tuning

### Batch Size

```python
# For XDNA (Phoenix/Hawk Point)
batch_size = 1  # Recommended

# For XDNA2 (Strix)
batch_size = 1-4  # Depends on model size
```

### Thread Configuration

```python
# Set number of parallel operations
npu_threads = 4  # Match to available tiles
```

### Memory Limits

```python
# Configure NPU memory limit
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.add_config_entry(
    'session.intra_op_num_threads', '4'
)
```

## Debugging

### Enable Logging

```bash
# XRT logging
export XRT_INI_PATH=/path/to/xrt.ini

# ONNX Runtime logging
export ORT_LOGGING_LEVEL=1
```

### Profile NPU Usage

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

session = ort.InferenceSession(
    "model.onnx",
    sess_options=sess_options
)

# Run inference...

# Get profile
profile = session.end_profiling()
print(profile)
```

## References

- [AMD XDNA Whitepaper](https://www.amd.com/en/developer/resources/technical-documents.html)
- [AIE-ML Programming Guide](https://docs.xilinx.com/r/en-US/ug1079-ai-engine-environment)
- [Vitis AI Documentation](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai)