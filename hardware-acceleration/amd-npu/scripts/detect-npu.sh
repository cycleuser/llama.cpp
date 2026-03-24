#!/bin/bash
# AMD NPU Hardware Detection Script
# Detects AMD XDNA NPU devices and checks driver status

set -e

echo "======================================"
echo "AMD NPU Hardware Detection"
echo "======================================"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

detect_cpu() {
    echo "[1] CPU Detection"
    echo "-----------------"
    
    if command -v lscpu &> /dev/null; then
        CPU_MODEL=$(lscpu | grep "Model name" | awk -F: '{print $2}' | xargs)
        echo "CPU: $CPU_MODEL"
        
        # Check if Ryzen AI capable
        if [[ "$CPU_MODEL" =~ "Ryzen" ]] && [[ "$CPU_MODEL" =~ ("7840"|"8840"|"7640"|"7940"|"8945"|"HX 370"|"365") ]]; then
            echo -e "${GREEN}✓ Ryzen AI compatible CPU detected${NC}"
            
            # Determine NPU generation
            if [[ "$CPU_MODEL" =~ ("7840"|"7640"|"7940") ]]; then
                echo "  NPU Architecture: XDNA (Phoenix)"
                echo "  NPU TOPS: 10"
                echo "  LLM Support: Hybrid (NPU + iGPU)"
            elif [[ "$CPU_MODEL" =~ ("8840"|"8945") ]]; then
                echo "  NPU Architecture: XDNA (Hawk Point)"
                echo "  NPU TOPS: 16"
                echo "  LLM Support: Hybrid (NPU + iGPU)"
            elif [[ "$CPU_MODEL" =~ ("HX 370"|"365"|"Max") ]]; then
                echo "  NPU Architecture: XDNA2 (Strix)"
                echo "  NPU TOPS: 50"
                echo "  LLM Support: Full NPU"
            fi
        else
            echo -e "${YELLOW}! CPU may not have AMD NPU${NC}"
        fi
    else
        echo -e "${RED}✗ lscpu not available${NC}"
    fi
    echo ""
}

detect_npu_pci() {
    echo "[2] NPU PCI Device Detection"
    echo "-----------------------------"
    
    # XDNA device IDs
    # Phoenix/Hawk Point: 0x1502
    # Strix: 0x17F0
    
    if command -v lspci &> /dev/null; then
        NPU_DEVICES=$(lspci -nn | grep -E "1022:(1502|17F0)" || true)
        
        if [ -n "$NPU_DEVICES" ]; then
            echo "$NPU_DEVICES"
            
            if echo "$NPU_DEVICES" | grep -q "1502"; then
                echo -e "${GREEN}✓ XDNA NPU (Phoenix/Hawk Point) detected${NC}"
                NPU_TYPE="XDNA"
            elif echo "$NPU_DEVICES" | grep -q "17F0"; then
                echo -e "${GREEN}✓ XDNA2 NPU (Strix) detected${NC}"
                NPU_TYPE="XDNA2"
            fi
        else
            echo -e "${YELLOW}! No AMD NPU PCI device found${NC}"
            echo "  Checking IPU devices..."
            
            # Alternative: IPU device path
            if [ -d "/sys/class/ipu" ]; then
                echo -e "${GREEN}✓ IPU device found in /sys/class/ipu${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ lspci not available${NC}"
        echo "  Install: sudo apt install pciutils"
    fi
    echo ""
}

detect_xdna_driver() {
    echo "[3] XDNA Driver Status"
    echo "----------------------"
    
    # Check kernel version
    KERNEL_VERSION=$(uname -r | cut -d. -f1,2)
    echo "Kernel version: $(uname -r)"
    
    # Check for amdxdna driver (kernel 6.14+)
    if lsmod | grep -q "amdxdna"; then
        echo -e "${GREEN}✓ amdxdna kernel module loaded${NC}"
    else
        echo -e "${YELLOW}! amdxdna kernel module not loaded${NC}"
        
        if [ -f "/lib/modules/$(uname -r)/kernel/drivers/accel/amdxdna/amdxdna.ko" ] || \
           [ -f "/lib/modules/$(uname -r)/extra/amdxdna.ko" ]; then
            echo "  Module exists, try: sudo modprobe amdxdna"
        else
            echo "  Module not found. Need to install xdna-driver"
        fi
    fi
    
    # Check device nodes
    if [ -d "/dev/accel" ]; then
        ACCEL_DEVICES=$(ls /dev/accel/ 2>/dev/null || true)
        if [ -n "$ACCEL_DEVICES" ]; then
            echo -e "${GREEN}✓ Accelerator devices: $ACCEL_DEVICES${NC}"
        fi
    fi
    
    # Check XRT
    if command -v xrt-smi &> /dev/null; then
        echo -e "${GREEN}✓ XRT installed${NC}"
        xrt-smi examine 2>/dev/null || true
    else
        echo -e "${YELLOW}! XRT not installed${NC}"
    fi
    echo ""
}

detect_gpu() {
    echo "[4] AMD GPU Detection (for hybrid execution)"
    echo "---------------------------------------------"
    
    if command -v lspci &> /dev/null; then
        AMD_GPU=$(lspci -nn | grep -E "VGA.*1022|Display.*1022" || true)
        
        if [ -n "$AMD_GPU" ]; then
            echo "$AMD_GPU"
            echo -e "${GREEN}✓ AMD GPU detected for hybrid execution${NC}"
        else
            echo -e "${YELLOW}! No AMD GPU detected${NC}"
            echo "  Hybrid NPU+iGPU execution requires AMD integrated graphics"
        fi
    fi
    echo ""
}

check_dependencies() {
    echo "[5] Software Dependencies"
    echo "-------------------------"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"
        
        # Check key packages
        python3 -c "import onnxruntime" 2>/dev/null && \
            echo -e "${GREEN}  ✓ onnxruntime installed${NC}" || \
            echo -e "${YELLOW}  ! onnxruntime not installed${NC}"
        
        python3 -c "import numpy" 2>/dev/null && \
            echo -e "${GREEN}  ✓ numpy installed${NC}" || \
            echo -e "${YELLOW}  ! numpy not installed${NC}"
    else
        echo -e "${RED}✗ Python3 not installed${NC}"
    fi
    
    # Check cmake
    if command -v cmake &> /dev/null; then
        echo -e "${GREEN}✓ CMake: $(cmake --version | head -1)${NC}"
    else
        echo -e "${YELLOW}! CMake not installed${NC}"
    fi
    echo ""
}

print_recommendations() {
    echo "======================================"
    echo "Recommendations"
    echo "======================================"
    
    if [ -z "$NPU_TYPE" ]; then
        echo "1. If you have a Ryzen AI CPU but NPU not detected:"
        echo "   - Update BIOS to enable IPU/NPU"
        echo "   - Check if NPU is disabled in BIOS"
        echo "   - Install latest chipset drivers"
        echo ""
    fi
    
    if ! lsmod | grep -q "amdxdna"; then
        echo "2. Install XDNA driver:"
        echo "   git clone https://github.com/amd/xdna-driver.git"
        echo "   cd xdna-driver && ./build.sh -release"
        echo "   sudo apt install ./Release/xrt_plugin.*.deb"
        echo ""
    fi
    
    if ! command -v xrt-smi &> /dev/null; then
        echo "3. Install XRT:"
        echo "   ./scripts/install-xrt.sh"
        echo ""
    fi
    
    echo "4. Build llama.cpp with NPU support:"
    echo "   cmake -B build -DGGML_AMD_NPU=ON"
    echo "   cmake --build build --config Release"
    echo ""
    
    echo "5. Run benchmark:"
    echo "   python3 tools/benchmark_npu.py"
}

# Main execution
detect_cpu
detect_npu_pci
detect_xdna_driver
detect_gpu
check_dependencies
print_recommendations