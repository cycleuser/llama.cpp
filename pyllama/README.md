# pyllama-server

Python wrapper for llama.cpp server - OpenAI API compatible backend with automatic device detection and model downloading.

## Features

- **Auto Device Detection**: Automatically detects and configures the best GPU backend (Vulkan, CUDA, ROCm, Metal)
- **Model Downloading**: Download GGUF models from HuggingFace and ModelScope
- **OpenAI API Compatible**: Drop-in replacement for OpenAI API
- **Function Calling**: Support for tools and function calling
- **Pre-built Binaries**: Automatically downloads pre-built llama.cpp binaries

## Installation

```bash
pip install pyllama-server
```

## Quick Start

### Command Line

```bash
# List available GPUs
pyllama devices

# Run inference with a model
pyllama run ./model.gguf -p "Hello, world!"

# Download a model
pyllama download llama-3.2-3b -q Q4_K_M

# Start an OpenAI-compatible server
pyllama serve llama-3.2-3b -p 8080
```

### Python API

```python
from pyllama import quick_run, quick_server, Client

# Quick inference
result = quick_run("llama-3.2-3b", "Write a haiku about coding")
print(result)

# Start server with auto-configuration
with quick_server("llama-3.2-3b") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0]["message"]["content"])
```

### Auto Device Detection

```python
from pyllama import DeviceDetector, AutoRunner

# Detect available GPUs
detector = DeviceDetector()
devices = detector.detect()
for device in devices:
    print(f"{device.name} ({device.backend.value}, {device.memory_gb:.1f}GB)")

# Get optimal configuration
config = detector.get_best_device(model_size_gb=5.0)
print(f"Best device: {config.device.name}")
print(f"Recommended GPU layers: {config.n_gpu_layers}")
```

### Model Download

```python
from pyllama import ModelDownloader

downloader = ModelDownloader()

# Download from HuggingFace
path = downloader.download("Qwen/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M.gguf")

# Download from ModelScope
path = downloader.download(
    "LLM-Research/Meta-Llama-3-8B-Instruct-GGUF",
    "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    source="modelscope"
)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `pyllama serve` | Start OpenAI-compatible server |
| `pyllama run` | Run inference with a model |
| `pyllama chat` | Interactive chat with a model |
| `pyllama download` | Download a model |
| `pyllama models` | List available models |
| `pyllama devices` | List GPU devices |
| `pyllama config` | Show optimal configuration |
| `pyllama download-binaries` | Download pre-built binaries |
| `pyllama build` | Build binaries from source |
| `pyllama clear-cache` | Clear model/binary cache |

## Popular Models

| Name | Description | Sizes |
|------|-------------|-------|
| llama-3.2-3b | Llama 3.2 3B | Q4_K_M, Q5_K_M, Q8_0 |
| llama-3.1-8b | Llama 3.1 8B | Q4_K_M, Q5_K_M, Q8_0 |
| qwen2.5-7b | Qwen 2.5 7B | Q4_K_M, Q5_K_M, Q8_0 |
| gemma-2-9b | Gemma 2 9B | Q4_K_M, Q5_K_M, Q8_0 |
| mistral-7b | Mistral 7B v0.3 | Q4_K_M, Q5_K_M, Q8_0 |
| phi-3.5-mini | Phi 3.5 Mini | Q4_K_M, Q5_K_M, Q8_0 |
| deepseek-coder-6.7b | DeepSeek Coder | Q4_K_M, Q5_K_M, Q8_0 |

## GPU Backends

| Backend | Platforms | Description |
|---------|-----------|-------------|
| Vulkan | Windows, Linux | Cross-platform GPU API, works on AMD, Intel, NVIDIA |
| CUDA | Windows, Linux | NVIDIA GPUs |
| ROCm | Linux | AMD GPUs |
| Metal | macOS | Apple Silicon |
| CPU | All | Fallback, no GPU required |

## Function Calling

```python
from pyllama import Client, LlamaServer

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

with LlamaServer("model.gguf") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools
    )
    
    if response.choices[0]["message"].get("tool_calls"):
        print("Model wants to call:", response.choices[0]["message"]["tool_calls"])
```

## Requirements

- Python 3.8+
- Vulkan SDK (for Vulkan backend)
- CUDA Toolkit (for CUDA backend)
- ROCm (for ROCm backend)

## License

MIT License - same as llama.cpp

## Links

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [Issues](https://github.com/ggml-org/llama.cpp/issues)