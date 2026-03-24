#!/bin/bash
# XRT (Xilinx Runtime) Installation Script for AMD NPU
# Supports Ubuntu 22.04/24.04

set -e

OS_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
echo "Detected OS: Ubuntu $OS_VERSION"

if [[ "$OS_VERSION" != "22.04" && "$OS_VERSION" != "24.04" ]]; then
    echo "Warning: This script is tested on Ubuntu 22.04 and 24.04"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "======================================"
echo "Installing XRT for AMD NPU"
echo "======================================"

# Install dependencies
echo "[1/5] Installing dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    rapidjson-dev \
    libdrm-dev \
    libpciaccess-dev \
    pkg-config \
    libelf-dev \
    dwb

# Install kernel headers
sudo apt install -y linux-headers-$(uname -r) || \
    sudo apt install -y linux-headers-generic

echo ""
echo "[2/5] Checking kernel version..."
KERNEL_MAJOR=$(uname -r | cut -d. -f1)
KERNEL_MINOR=$(uname -r | cut -d. -f2)

if [ "$KERNEL_MAJOR" -lt 6 ] || ([ "$KERNEL_MAJOR" -eq 6 ] && [ "$KERNEL_MINOR" -lt 10 ]); then
    echo "Kernel $(uname -r) may not have native amdxdna support"
    echo "Consider upgrading to kernel 6.14+:"
    echo "  sudo apt install linux-generic-hwe-24.04"
fi

echo ""
echo "[3/5] Downloading XDNA driver..."
cd /tmp
if [ -d "xdna-driver" ]; then
    echo "xdna-driver already exists, updating..."
    cd xdna-driver
    git pull
else
    git clone --recursive https://github.com/amd/xdna-driver.git
    cd xdna-driver
fi

echo ""
echo "[4/5] Building XDNA driver and XRT plugin..."
mkdir -p build
cd build

# Build
../build.sh -release

echo ""
echo "[5/5] Installing..."

# Install XRT base (if needed)
if [ ! -f "./Release/xrt_*_amd64.deb" ]; then
    echo "XRT package not found, downloading from Xilinx..."
    XRT_VERSION="2.18.0"
    XRT_URL="https://www.xilinx.com/bin/public/openDownload?filename=xrt_${XRT_VERSION}_$(lsb_release -cs)_amd64.deb"
    
    if command -v wget &> /dev/null; then
        wget -O xrt.deb "$XRT_URL" || {
            echo "Failed to download XRT from Xilinx"
            echo "Installing from package manager..."
            sudo apt install -y xrt 2>/dev/null || echo "XRT not available in package manager"
        }
    fi
fi

# Install XRT plugin for amdxdna
if ls ./Release/xrt_plugin.*.deb 1> /dev/null 2>&1; then
    echo "Installing XRT plugin..."
    sudo apt install -y ./Release/xrt_plugin.*.deb
else
    echo "Warning: XRT plugin package not found"
fi

echo ""
echo "======================================"
echo "Installation Complete"
echo "======================================"

# Verify installation
if command -v xrt-smi &> /dev/null; then
    echo "XRT version: $(xrt-smi --version 2>/dev/null | head -1 || echo 'unknown')"
    echo ""
    echo "Detecting NPU devices..."
    xrt-smi examine 2>/dev/null || echo "No NPU devices detected yet"
else
    echo "xrt-smi not found in PATH"
    echo "You may need to restart your terminal or run:"
    echo "  source /opt/xilinx/xrt/setup.sh"
fi

echo ""
echo "Next steps:"
echo "1. Reboot or reload kernel modules: sudo modprobe amdxdna"
echo "2. Verify NPU: python3 scripts/verify_npu.py"
echo "3. Build llama.cpp: cmake -B build -DGGML_AMD_NPU=ON"