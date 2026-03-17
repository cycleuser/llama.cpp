#!/bin/bash
#
# Device Detection and Benchmark Script
# Detects all available devices and runs benchmarks
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

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

detect_cpu() {
    print_header "CPU Information"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Platform: $(sysctl -n machdep.cpu.brand_string)"
        echo "Cores: $(sysctl -n hw.ncpu)"
        echo "Memory: $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB"
    else
        if command -v lscpu &> /dev/null; then
            echo "Platform: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
            echo "Cores: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
        fi
        if [[ -f /proc/meminfo ]]; then
            echo "Memory: $(awk '/MemTotal/ {printf "%.0f GB", $2/1024/1024}' /proc/meminfo)"
        fi
    fi
    echo ""
}

detect_apple_silicon() {
    if [[ "$OSTYPE" != "darwin"* ]]; then
        return
    fi
    
    print_header "Apple Silicon GPU"
    
    if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple M"; then
        local chip=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | cut -d: -f2 | xargs)
        local cores=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Total Number of Cores" | head -1 | cut -d: -f2 | xargs)
        local metal=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Metal" | head -1 | cut -d: -f2 | xargs)
        
        echo "Chip: $chip"
        echo "GPU Cores: $cores"
        echo "Metal Version: $metal"
        print_success "Metal backend available"
    else
        print_info "No Apple Silicon GPU detected"
    fi
    echo ""
}

detect_nvidia() {
    print_header "NVIDIA GPU"
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while IFS=',' read -r name mem driver; do
            echo "GPU: $(echo $name | xargs)"
            echo "Memory: $(echo $mem | xargs)"
            echo "Driver: $(echo $driver | xargs)"
        done
        print_success "CUDA backend available"
    else
        print_info "No NVIDIA GPU detected or nvidia-smi not found"
    fi
    echo ""
}

detect_amd() {
    print_header "AMD GPU"
    
    if command -v rocminfo &> /dev/null; then
        rocminfo 2>/dev/null | grep -A 10 "GPU" | head -20
        print_success "ROCm/HIP backend available"
    elif command -v clinfo &> /dev/null; then
        local amd_devices=$(clinfo -l 2>/dev/null | grep -i "amd\|radeon" || true)
        if [[ -n "$amd_devices" ]]; then
            echo "$amd_devices"
            print_success "OpenCL backend available for AMD"
        else
            print_info "No AMD GPU detected via OpenCL"
        fi
    else
        print_info "No AMD GPU detection tools found (rocminfo, clinfo)"
    fi
    echo ""
}

detect_intel() {
    print_header "Intel GPU"
    
    if command -v clinfo &> /dev/null; then
        local intel_devices=$(clinfo -l 2>/dev/null | grep -i "intel" || true)
        if [[ -n "$intel_devices" ]]; then
            echo "$intel_devices"
            print_success "OpenCL/SYCL backend available for Intel"
        else
            print_info "No Intel GPU detected"
        fi
    else
        print_info "clinfo not found, skipping Intel GPU detection"
    fi
    echo ""
}

detect_vulkan() {
    print_header "Vulkan Devices"
    
    if command -v vulkaninfo &> /dev/null; then
        vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion|apiVersion" | head -20
        print_success "Vulkan backend available"
    else
        print_info "vulkaninfo not found, install vulkan-tools"
    fi
    echo ""
}

run_llama_bench() {
    local model="$1"
    local device="$2"
    local n_threads="$3"
    local n_gpu_layers="$4"
    
    if [[ ! -f "$model" ]]; then
        print_error "Model not found: $model"
        return 1
    fi
    
    local llama_cli="$BUILD_DIR/bin/llama-cli"
    if [[ ! -x "$llama_cli" ]]; then
        print_error "llama-cli not found. Run build first."
        return 1
    fi
    
    local args=()
    args+=("-m" "$model")
    args+=("-p" "Hello, how are you today?")
    args+=("-n" "64")
    args+=("-t" "$n_threads")
    
    case "$device" in
        cpu)
            args+=("-ngl" "0")
            ;;
        metal)
            args+=("-ngl" "$n_gpu_layers")
            ;;
        cuda|hip)
            args+=("-ngl" "$n_gpu_layers")
            ;;
        vulkan)
            args+=("-ngl" "$n_gpu_layers")
            ;;
    esac
    
    echo "Running: $llama_cli ${args[*]}"
    "$llama_cli" "${args[@]}" 2>&1 | tail -5
}

print_summary() {
    print_header "Device Summary"
    
    echo ""
    echo "Recommended commands:"
    echo ""
    
    if [[ "$OSTYPE" == "darwin"* ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Apple M"; then
        echo "# Build for Apple Metal (M-series):"
        echo "cmake -B build && cmake --build build -j"
        echo ""
        echo "# Run inference:"
        echo "./build/bin/llama-cli -m model.gguf -p \"Hello\" -ngl 99"
        echo ""
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        echo "# Build for NVIDIA CUDA:"
        echo "cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
        echo ""
        echo "# Run inference:"
        echo "./build/bin/llama-cli -m model.gguf -p \"Hello\" -ngl 99"
        echo ""
    fi
    
    if command -v rocminfo &> /dev/null; then
        echo "# Build for AMD ROCm/HIP:"
        echo "cmake -B build -DGGML_HIP=ON && cmake --build build -j"
        echo ""
        echo "# For older AMD GPUs (RX580/RX590), also set:"
        echo "export HSA_OVERRIDE_GFX_VERSION=8.0.3"
        echo ""
    fi
    
    echo "# Build with Vulkan (cross-platform):"
    echo "cmake -B build -DGGML_VULKAN=ON && cmake --build build -j"
    echo ""
    
    echo "# Multi-device parallel inference:"
    echo "./hardware-acceleration/tools/multi-device-runner -m model1.gguf -d cpu -m model2.gguf -d gpu0"
    echo ""
}

main() {
    echo ""
    print_header "Hardware Detection for llama.cpp"
    echo ""
    
    detect_cpu
    detect_apple_silicon
    detect_nvidia
    detect_amd
    detect_intel
    detect_vulkan
    
    print_summary
}

main "$@"