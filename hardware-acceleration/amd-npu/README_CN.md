# AMD NPU Support - 中文文档

AMD Ryzen AI NPU (XDNA/XDNA2) 的全面支持方案。

## 硬件支持列表

### XDNA 架构（第一代）- 混合执行

| 处理器 | NPU TOPS | 内存 | LLM 支持 |
|--------|----------|------|----------|
| Ryzen 5 7640HS | 10 | ~2GB | 混合模式 |
| Ryzen 7 7840HS | 10 | ~2GB | 混合模式 |
| Ryzen 7 8840HS | 16 | ~2GB | 混合模式 |
| Ryzen 9 7940HS | 10 | ~2GB | 混合模式 |
| Ryzen 9 8945HS | 16 | ~2GB | 混合模式 |

### XDNA2 架构（第二代）- 完整 NPU 执行

| 处理器 | NPU TOPS | 内存 | LLM 支持 |
|--------|----------|------|----------|
| Ryzen AI 9 HX 370 | 50 | ~4GB | 完整 NPU |
| Ryzen AI 9 365 | 50 | ~4GB | 完整 NPU |
| Ryzen AI Max+ 395 | 75 | ~8GB | 完整 NPU |

## 快速开始

### Windows 安装

```powershell
# 1. 安装 NPU 驱动（Windows Update 自动安装）

# 2. 安装 Ryzen AI 软件
# 下载: https://www.amd.com/en/developer/resources/ryzen-ai.html

# 3. 安装 Python 依赖
pip install numpy onnx onnxruntime

# 4. 验证安装
python scripts\verify_npu.py
```

### Linux 安装

```bash
# 1. 安装内核 6.14+
sudo apt install linux-generic-hwe-24.04

# 2. 安装 XDNA 驱动
./scripts/install-xrt.sh

# 3. 验证安装
python3 scripts/verify_npu.py
```

### 构建 llama.cpp

```bash
# 启用 AMD NPU 支持
cmake -B build -DGGML_AMD_NPU=ON
cmake --build build --config Release

# 运行推理
./build/bin/llama-cli -m model.gguf -p "你好"
```

## 模型转换

```bash
# GGUF 转 ONNX
python tools/gguf_to_onnx.py model.gguf -o model.onnx

# 编译为 NPU 优化格式
python tools/compile_model.py model.onnx -t PHX  # Phoenix
python tools/compile_model.py model.onnx -t STX  # Strix

# INT8 量化
python tools/gguf_to_onnx.py model.gguf --quantize
```

## 性能基准

| 模型 | Phoenix (混合) | Hawk Point (混合) | Strix (NPU) |
|------|----------------|-------------------|-------------|
| Llama-3.2-1B | ~20 t/s | ~25 t/s | ~40 t/s |
| Llama-3.2-3B | ~8 t/s | ~12 t/s | ~25 t/s |
| Qwen-2.5-1.5B | ~22 t/s | ~28 t/s | ~45 t/s |
| Phi-3-mini | ~15 t/s | ~20 t/s | ~35 t/s |

## 混合执行模式

对于 Phoenix/Hawk Point，使用 NPU + iGPU 混合执行：

```
┌──────────────────────────────────────┐
│          混合执行架构                 │
├──────────────────────────────────────┤
│  Prompt 处理  →  NPU (注意力计算)    │
│  Token 生成   →  iGPU (解码层)       │
│  嵌入层       →  CPU                 │
└──────────────────────────────────────┘
```

配置：
```bash
# 启用混合模式
export GGML_AMD_NPU_HYBRID=1

# 设置 NPU 层数
./llama-cli -m model.gguf -ngl 16 --split-mode layer
```

## 目录结构

```
amd-npu/
├── src/                    # 源代码
│   ├── ggml-amdxdna.cpp   # 后端实现
│   └── ggml-amdxdna.h     # 头文件
├── tools/                  # 工具
│   ├── gguf_to_onnx.py    # 模型转换
│   ├── compile_model.py   # NPU 编译
│   └── benchmark_npu.py   # 性能测试
├── scripts/                # 脚本
│   ├── detect-npu.sh      # 硬件检测
│   ├── install-xrt.sh     # XRT 安装
│   └── verify_npu.py      # 验证安装
├── docs/                   # 文档
│   ├── installation.md    # 安装指南
│   ├── architecture.md    # 架构说明
│   └── performance.md     # 性能优化
└── tests/                  # 测试
    └── test_amdxdna.cpp   # 单元测试
```

## 已知限制

1. **Phoenix/Hawk Point**: LLM 需要混合执行
2. **内存**: NPU 内存有限，大模型需要量化
3. **量化**: INT8 模型在 NPU 上性能最佳
4. **驱动**: Windows 支持比 Linux 更成熟

## 故障排除

### NPU 未检测到

```bash
# 检查 BIOS 设置
# 确保 IPU/NPU 已启用

# 检查驱动
lspci -nn | grep -E "1022:(1502|17F0)"

# 加载内核模块
sudo modprobe amdxdna
```

### 性能不佳

```bash
# 检查是否使用 NPU
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 使用 INT8 量化模型
python tools/gguf_to_onnx.py model.gguf --quantize
```

### 内存不足

```bash
# 减小 batch size
# 使用更小的上下文长度
# 启用内存卸载
export GGML_AMD_NPU_MEMORY_LIMIT=1073741824  # 1GB
```

## 参考链接

- [Ryzen AI 文档](https://ryzenai.docs.amd.com)
- [XDNA 驱动](https://github.com/amd/xdna-driver)
- [AMD whisper.cpp](https://github.com/amd/whisper.cpp)
- [RyzenAI-SW 示例](https://github.com/amd/RyzenAI-SW)