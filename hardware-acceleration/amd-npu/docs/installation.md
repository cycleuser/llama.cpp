# AMD NPU Installation Guide

Complete installation guide for AMD XDNA NPU support on Windows and Linux.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Windows Installation](#windows-installation)
3. [Linux Installation](#linux-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

## Hardware Requirements

### Supported Processors

**XDNA (Gen 1) - Hybrid NPU+iGPU Execution**
- Ryzen 5 7640HS / 7640U
- Ryzen 7 7840HS / 7840U / 7840H
- Ryzen 9 7940HS / 7940H
- Ryzen 7 8840HS / 8840U / 8840H (Hawk Point)
- Ryzen 9 8945HS / 8945H (Hawk Point)

**XDNA2 (Gen 2) - Full NPU Execution**
- Ryzen AI 9 HX 370
- Ryzen AI 9 365
- Ryzen AI Max+ 395 (Strix Halo)
- Ryzen AI 7 PRO 360 (Krackan)

### BIOS Settings

Ensure the following are enabled in BIOS:
- IPU (Image Processing Unit) / NPU
- SVM (AMD-V) virtualization
- Above 4G Decoding

## Windows Installation

### Step 1: Update BIOS and Drivers

```powershell
# Check Windows version (requires Windows 11 23H2 or later)
winver

# Update Windows to get latest drivers
# Settings > Windows Update > Check for updates
```

### Step 2: Install NPU Driver

1. Download NPU driver from AMD:
   - [AMD NPU Driver Downloads](https://www.amd.com/en/support)
   - Or use Windows Update (recommended)

2. Install the driver:
   ```powershell
   # Run the downloaded installer
   # Or use Device Manager to update driver
   ```

3. Verify in Device Manager:
   - Look for "AMD IPU Device" or "AMD NPU" under "Processing devices"

### Step 3: Install Ryzen AI Software

1. Download Ryzen AI Software:
   - [Ryzen AI Software 1.7.0+](https://www.amd.com/en/developer/resources/ryzen-ai.html)

2. Install with default options:
   ```powershell
   # Run the installer
   ryzen-ai-lt-1.7.0.exe
   ```

3. Set environment variable:
   ```powershell
   # Add to system PATH or set manually
   $env:RYZEN_AI_INSTALLATION_PATH = "C:\Program Files\AMD\RyzenAI"
   ```

### Step 4: Install Python Dependencies

```powershell
# Install Python 3.10+ if not already installed
winget install Python.Python.3.12

# Install required packages
pip install numpy onnx onnxruntime
pip install ryzen-ai-libraries  # From Ryzen AI installation
```

### Step 5: Verify Installation

```powershell
# Run verification script
python scripts\verify_npu.py

# Or run quick test
cd %RYZEN_AI_INSTALLATION_PATH%\quicktest
python quicktest.py
```

## Linux Installation

### Step 1: Install Required Kernel

```bash
# Ubuntu 24.04+ with HWE kernel
sudo apt update
sudo apt install linux-generic-hwe-24.04

# Or compile kernel 6.14+ with amdxdna driver
```

### Step 2: Install Build Dependencies

```bash
sudo apt install -y \
    build-essential cmake git \
    python3 python3-pip python3-dev \
    libboost-all-dev \
    libprotobuf-dev protobuf-compiler \
    rapidjson-dev \
    libdrm-dev libpciaccess-dev \
    pkg-config libelf-dev \
    linux-headers-$(uname -r)
```

### Step 3: Build and Install XDNA Driver

```bash
# Clone the driver
git clone --recursive https://github.com/amd/xdna-driver.git
cd xdna-driver

# Build
mkdir build && cd build
../build.sh -release

# Install XRT plugin
sudo apt install ./Release/xrt_plugin.*.deb
```

### Step 4: Install XRT

```bash
# Option A: From AMD repository
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_2.18.0_ubuntu22.04_amd64.deb
sudo apt install ./xrt_*.deb

# Option B: From source
git clone https://github.com/Xilinx/XRT.git
cd XRT && ./build.sh
sudo ./build/Release/xrt_*_amd64.deb
```

### Step 5: Load Kernel Module

```bash
# Load amdxdna module
sudo modprobe amdxdna

# Verify
lsmod | grep amdxdna
dmesg | grep -i amdxdna

# Check device nodes
ls -la /dev/accel/
```

### Step 6: Install Python Dependencies

```bash
pip3 install numpy onnx onnxruntime
```

### Step 7: Verify Installation

```bash
# Check XRT
xrt-smi examine

# Run verification script
python3 scripts/verify_npu.py
```

## Verification

### Quick Test

```bash
# Linux
python3 scripts/verify_npu.py

# Windows
python scripts\verify_npu.py
```

### Expected Output

```
[1] CPU Detection
-----------------
CPU: AMD Ryzen 7 7840HS with Radeon Graphics
✓ Ryzen AI compatible CPU detected
  NPU Architecture: XDNA (Phoenix)
  NPU TOPS: 10
  LLM Support: Hybrid (NPU + iGPU)

[2] NPU PCI Device Detection
-----------------------------
04:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD] XDNA
✓ XDNA NPU (Phoenix/Hawk Point) detected

[3] XDNA Driver Status
----------------------
✓ amdxdna kernel module loaded
✓ Accelerator devices: accel0
✓ XRT installed
```

## Troubleshooting

### NPU Not Detected

1. **Check BIOS settings**
   - Ensure IPU/NPU is enabled
   - Update to latest BIOS version

2. **Check driver installation**
   ```bash
   # Linux
   lspci -nn | grep -E "1022:(1502|17F0)"
   lsmod | grep amdxdna
   
   # Windows
   # Device Manager > Processing devices
   ```

3. **Reinstall drivers**
   ```bash
   # Linux
   sudo modprobe -r amdxdna
   sudo modprobe amdxdna
   
   # Windows
   # Uninstall from Device Manager, then reinstall
   ```

### XRT Errors

1. **xrt-smi not found**
   ```bash
   source /opt/xilinx/xrt/setup.sh
   ```

2. **Permission denied**
   ```bash
   sudo usermod -a -G render,video $USER
   # Log out and back in
   ```

3. **Version mismatch**
   ```bash
   xrt-smi --version
   # Ensure XRT version matches driver version
   ```

### Performance Issues

1. **Slow inference**
   - Ensure NPU is being used (check provider)
   - Use INT8 quantized models
   - Enable hybrid mode for Phoenix/Hawk Point

2. **Out of memory**
   - Reduce batch size
   - Use smaller model
   - Enable memory limits

## Additional Resources

- [Ryzen AI Documentation](https://ryzenai.docs.amd.com)
- [XDNA Driver GitHub](https://github.com/amd/xdna-driver)
- [AMD Developer Forums](https://community.amd.com/t5/developer-forums/bd-p/developer-discussion-forum)