#!/bin/bash
# =============================================================================
# Hardware-Optimized Build Script for llama.cpp
# =============================================================================
# 
# Auto-detects available hardware and builds the optimal llama.cpp version.
# 
# Usage:
#   ./build-hardware-optimized.sh           # Auto-detect
#   ./build-hhardware-optimized.sh metal    # Apple Metal
#   ./build-hardware-optimized.sh hip       # AMD ROCm/HIP
#   ./build-hardware-optimized.sh cuda      # NVIDIA CUDA
#   ./build-hardware-optimized.sh vulkan    # Vulkan
#   ./build-hardware-optimized.sh sycl      # Intel SYCL
#   ./build-hardware-optimized.sh opencl    # OpenCL
#   ./build-hardware-optimized.sh all       # Build all backends
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "Linux" ;;
        Darwin*) echo "macOS" ;;
        CYGWIN*|MINGW*|MSYS*) echo "Windows" ;;
        *)       echo "Unknown" ;;
    esac
}

detect_apple_silicon() {
    if [[ "$(uname -m)" == "arm64" ]]; then
        return 0
    fi
    return 1
}

detect_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi &> /dev/null && return 0
    fi
    return 1
}

detect_amd_gpu() {
    if command -v rocminfo &> /dev/null; then
        rocminfo | grep -q "gfx" && return 0
    fi
    if command -v vulkaninfo &> /dev/null; then
        vulkaninfo | grep -q "AMD" && return 0
    fi
    return 1
}

detect_intel_gpu() {
    if command -v clinfo &> /dev/null; then
        clinfo | grep -q "Intel" && return 0
    fi
    return 1
}

build_metal() {
    echo -e "${CYAN}Building with Metal backend for Apple Silicon...${NC}"
    cmake -B build-metal -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-metal --config Release -j
    echo -e "${GREEN}Metal build complete: build-metal/bin/${NC}"
}

build_cuda() {
    echo -e "${CYAN}Building with CUDA backend for NVIDIA GPUs...${NC}"
    cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cuda --config Release -j
    echo -e "${GREEN}CUDA build complete: build-cuda/bin/${NC}"
}

build_hip() {
    echo -e "${CYAN}Building with HIP backend for AMD GPUs...${NC}"
    
    # Check for ROCm
    if [[ ! -d "/opt/rocm" && ! -d "$HOME/rocm" ]]; then
        echo -e "${YELLOW}Warning: ROCm not found. Make sure it's installed.${NC}"
    fi
    
    cmake -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc
    cmake --build build-hip --config Release -j
    echo -e "${GREEN}HIP build complete: build-hip/bin/${NC}"
}

build_vulkan() {
    echo -e "${CYAN}Building with Vulkan backend...${NC}"
    cmake -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-vulkan --config Release -j
    echo -e "${GREEN}Vulkan build complete: build-vulkan/bin/${NC}"
}

build_sycl() {
    echo -e "${CYAN}Building with SYCL backend for Intel GPUs...${NC}"
    cmake -B build-sycl -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-sycl --config Release -j
    echo -e "${GREEN}SYCL build complete: build-sycl/bin/${NC}"
}

build_opencl() {
    echo -e "${CYAN}Building with OpenCL backend...${NC}"
    cmake -B build-opencl -DGGML_OPENCL=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-opencl --config Release -j
    echo -e "${GREEN}OpenCL build complete: build-opencl/bin/${NC}"
}

build_cpu() {
    echo -e "${CYAN}Building CPU-only version...${NC}"
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j
    echo -e "${GREEN}CPU build complete: build/bin/${NC}"
}

build_all() {
    local os=$(detect_os)
    
    case "$os" in
        macOS)
            detect_apple_silicon && build_metal
            ;;
        Linux)
            detect_nvidia_gpu && build_cuda
            detect_amd_gpu && build_hip
            build_vulkan
            detect_intel_gpu && build_sycl
            build_opencl
            ;;
        *)
            build_vulkan
            build_opencl
            ;;
    esac
    
    build_cpu
}

auto_build() {
    local os=$(detect_os)
    echo -e "${CYAN}Detected OS: $os${NC}"
    
    case "$os" in
        macOS)
            if detect_apple_silicon; then
                echo -e "${GREEN}Apple Silicon detected${NC}"
                build_metal
            else
                build_cpu
            fi
            ;;
        Linux)
            if detect_nvidia_gpu; then
                echo -e "${GREEN}NVIDIA GPU detected${NC}"
                build_cuda
            elif detect_amd_gpu; then
                echo -e "${GREEN}AMD GPU detected${NC}"
                build_hip || build_vulkan
            elif detect_intel_gpu; then
                echo -e "${GREEN}Intel GPU detected${NC}"
                build_sycl || build_opencl
            else
                build_cpu
            fi
            ;;
        *)
            build_cpu
            ;;
    esac
}

main() {
    cd "$ROOT_DIR"
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Hardware-Optimized llama.cpp Builder${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    local backend="${1:-auto}"
    
    case "$backend" in
        metal)    build_metal ;;
        cuda)     build_cuda ;;
        hip)      build_hip ;;
        vulkan)   build_vulkan ;;
        sycl)     build_sycl ;;
        opencl)   build_opencl ;;
        cpu)      build_cpu ;;
        all)      build_all ;;
        auto|*)   auto_build ;;
    esac
}

main "$@"