#!/bin/bash
#
# Performance Benchmark Script
# Runs comprehensive benchmarks across all available devices
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

DEFAULT_MODEL="${MODEL_PATH:-}"
DEFAULT_PROMPT="Write a short story about a robot learning to paint."
DEFAULT_TOKENS=128
DEFAULT_THREADS=4

print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

run_bench() {
    local model="$1"
    local device="$2"
    local ngls="$3"
    local threads="$4"
    local tokens="$5"
    local prompt="$6"
    
    local llama_cli="$BUILD_DIR/bin/llama-cli"
    if [[ ! -x "$llama_cli" ]]; then
        echo -e "${RED}Error: llama-cli not found at $llama_cli${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Device: $device, GPU layers: $ngls${NC}"
    
    local output
    output=$("$llama_cli" \
        -m "$model" \
        -p "$prompt" \
        -n "$tokens" \
        -t "$threads" \
        -ngl "$ngls" \
        --no-display-prompt \
        2>&1)
    
    local tps=$(echo "$output" | grep -oP 'eval time.*?(\d+\.\d+) tokens per second' | grep -oP '\d+\.\d+' | tail -1)
    
    if [[ -n "$tps" ]]; then
        echo -e "${GREEN}Speed: $tps tokens/sec${NC}"
    else
        echo -e "${RED}Failed to parse tokens/sec${NC}"
    fi
    
    echo ""
}

run_parallel_bench() {
    local model="$1"
    local threads="$2"
    local tokens="$3"
    local prompt="$4"
    
    local runner="$PROJECT_ROOT/hardware-acceleration/tools/multi-device-runner"
    if [[ ! -x "$runner" ]]; then
        echo -e "${YELLOW}Building multi-device-runner...${NC}"
        cd "$PROJECT_ROOT"
        g++ -O3 -std=c++17 \
            -I. -I./src -I./ggml/include \
            hardware-acceleration/tools/multi-device-runner.cpp \
            -o "$runner" \
            -L./build/src -lllama \
            -L./build/ggml/src -lggml \
            -L./build/ggml/src/ggml-cpu -lggml-cpu \
            -lpthread -ldl 2>/dev/null || {
            echo -e "${RED}Failed to build multi-device-runner${NC}"
            return 1
        }
    fi
    
    print_header "Parallel Multi-Device Benchmark"
    
    "$runner" -m "$model" -m "$model" -d cpu -d gpu0 \
        -p "$prompt" -n "$tokens" -t "$threads" 2>&1 || true
}

main() {
    local model="${1:-$DEFAULT_MODEL}"
    
    if [[ -z "$model" ]]; then
        echo "Usage: $0 <model.gguf>"
        echo ""
        echo "Or set MODEL_PATH environment variable:"
        echo "  MODEL_PATH=model.gguf $0"
        exit 1
    fi
    
    if [[ ! -f "$model" ]]; then
        echo -e "${RED}Model not found: $model${NC}"
        exit 1
    fi
    
    print_header "Performance Benchmark"
    echo "Model: $model"
    echo "Prompt: $DEFAULT_PROMPT"
    echo "Tokens to generate: $DEFAULT_TOKENS"
    
    print_header "CPU Benchmark (all layers on CPU)"
    run_bench "$model" "CPU" 0 "$DEFAULT_THREADS" "$DEFAULT_TOKENS" "$DEFAULT_PROMPT"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple M"; then
            print_header "Metal Benchmark (GPU offload)"
            run_bench "$model" "Metal" 99 "$DEFAULT_THREADS" "$DEFAULT_TOKENS" "$DEFAULT_PROMPT"
        fi
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        print_header "CUDA Benchmark (GPU offload)"
        run_bench "$model" "CUDA" 99 "$DEFAULT_THREADS" "$DEFAULT_TOKENS" "$DEFAULT_PROMPT"
    fi
    
    if command -v vulkaninfo &> /dev/null; then
        print_header "Vulkan Benchmark (GPU offload)"
        run_bench "$model" "Vulkan" 99 "$DEFAULT_THREADS" "$DEFAULT_TOKENS" "$DEFAULT_PROMPT"
    fi
    
    print_header "Summary"
    echo "To run parallel inference on multiple devices:"
    echo "  ./hardware-acceleration/tools/multi-device-runner -m $model -d cpu -m $model -d gpu0"
}

main "$@"