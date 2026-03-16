#!/bin/bash
# =============================================================================
# Hardware Acceleration Test Script
# =============================================================================
#
# Tests all available hardware acceleration backends.
#
# Usage:
#   ./test.sh              # Quick test
#   ./test.sh --full       # Full test with benchmarks
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

test_metal() {
    local bin="$ROOT_DIR/build-metal/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] Metal backend available${NC}"
        "$bin" --version 2>&1 | grep -E "Metal|GPU|simdgroup" || true
        return 0
    else
        echo -e "${YELLOW}[--] Metal backend not built${NC}"
        return 1
    fi
}

test_cuda() {
    local bin="$ROOT_DIR/build-cuda/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] CUDA backend available${NC}"
        "$bin" --version 2>&1 | grep -E "CUDA|GPU" || true
        return 0
    else
        echo -e "${YELLOW}[--] CUDA backend not built${NC}"
        return 1
    fi
}

test_hip() {
    local bin="$ROOT_DIR/build-hip/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] HIP backend available${NC}"
        "$bin" --version 2>&1 | grep -E "HIP|ROCm|GPU" || true
        return 0
    else
        echo -e "${YELLOW}[--] HIP backend not built${NC}"
        return 1
    fi
}

test_vulkan() {
    local bin="$ROOT_DIR/build-vulkan/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] Vulkan backend available${NC}"
        "$bin" --version 2>&1 | grep -E "Vulkan|GPU" || true
        return 0
    else
        echo -e "${YELLOW}[--] Vulkan backend not built${NC}"
        return 1
    fi
}

test_sycl() {
    local bin="$ROOT_DIR/build-sycl/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] SYCL backend available${NC}"
        "$bin" --version 2>&1 | grep -E "SYCL|GPU" || true
        return 0
    else
        echo -e "${YELLOW}[--] SYCL backend not built${NC}"
        return 1
    fi
}

test_opencl() {
    local bin="$ROOT_DIR/build-opencl/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] OpenCL backend available${NC}"
        "$bin" --version 2>&1 | grep -E "OpenCL|GPU" || true
        return 0
    else
        echo -e "${YELLOW}[--] OpenCL backend not built${NC}"
        return 1
    fi
}

test_cpu() {
    local bin="$ROOT_DIR/build/bin/llama-cli"
    if [ -x "$bin" ]; then
        echo -e "${GREEN}[OK] CPU backend available${NC}"
        "$bin" --version 2>&1 | head -5
        return 0
    else
        echo -e "${RED}[XX] CPU backend not built${NC}"
        return 1
    fi
}

main() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}Hardware Acceleration Test${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    local os=$(uname -s)
    echo -e "${CYAN}OS: $os${NC}"
    echo ""
    
    echo -e "${CYAN}Testing backends...${NC}"
    echo ""
    
    case "$os" in
        Darwin)
            test_metal
            ;;
        Linux)
            test_cuda
            test_hip
            test_vulkan
            test_sycl
            test_opencl
            ;;
    esac
    
    test_vulkan
    test_cpu
    
    echo ""
    echo -e "${CYAN}Test complete.${NC}"
}

main "$@"