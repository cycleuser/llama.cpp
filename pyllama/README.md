# pyllama-server

Python wrapper for llama.cpp server with OpenAI API compatibility.

## Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI API
- **Multi-GPU Support**: Vulkan, CUDA, ROCm, Metal backends
- **Function Calling**: Support for tools/function calling
- **Multimodal**: Vision and audio model support
- **Streaming**: Real-time token streaming
- **Easy Installation**: pip install and go

## Installation

```bash
pip install pyllama-server
```

## Quick Start

### Start a server

```bash
pyllama serve -m model.gguf -p 8080
```

### Chat with a model

```python
from pyllama import Client, LlamaServer

# Start server with context manager
with LlamaServer(model="model.gguf") as server:
    client = Client(server.base_url)
    
    response = client.chat.completions.create(
        model="llama",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(response.choices[0]["message"]["content"])
```

### OpenAI SDK compatible

```python
from openai import OpenAI
from pyllama import LlamaServer

with LlamaServer(model="model.gguf", port=8080):
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="dummy"
    )
    
    response = client.chat.completions.create(
        model="llama",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### Function Calling / Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

response = client.chat.completions.create(
    model="llama",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)
```

## CLI Commands

```bash
# Start server
pyllama serve -m model.gguf

# List GPUs
pyllama gpus

# Quick inference
pyllama run model.gguf -p "Hello"

# Build binaries
pyllama build --backend vulkan
```

## GPU Backend Support

| Backend | AMD | NVIDIA | Apple | Intel |
|---------|-----|--------|-------|-------|
| Vulkan  | ✅   | ✅      | ✅     | ✅     |
| CUDA    | ❌   | ✅      | ❌     | ❌     |
| ROCm    | ✅   | ❌      | ❌     | ❌     |
| Metal   | ❌   | ❌      | ✅     | ❌     |

For AMD RX580/RX590, use **Vulkan** backend.

## Building from Source

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build with Vulkan (recommended for AMD)
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j8

# Install Python package
cd pyllama
pip install -e .
```

## License

MIT