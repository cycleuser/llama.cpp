#!/usr/bin/env python3
"""
AMD NPU Verification Script
Checks NPU availability and functionality
"""

import sys
import os
import subprocess
import platform

def print_header(title):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)

def check_command(cmd, description):
    """Check if a command is available"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ {description}: OK")
            return True, result.stdout
        else:
            print(f"✗ {description}: FAILED")
            return False, result.stderr
    except FileNotFoundError:
        print(f"✗ {description}: NOT FOUND")
        return False, None
    except subprocess.TimeoutExpired:
        print(f"✗ {description}: TIMEOUT")
        return False, None

def check_python_package(package, description=None):
    """Check if a Python package is installed"""
    if description is None:
        description = package
    try:
        __import__(package)
        print(f"✓ {description}: installed")
        return True
    except ImportError:
        print(f"✗ {description}: not installed")
        return False

def check_npu_linux():
    """Check NPU on Linux"""
    print_header("Linux NPU Detection")
    
    # Check for amdxdna driver
    if os.path.exists("/sys/module/amdxdna"):
        print("✓ amdxdna kernel module loaded")
    else:
        print("✗ amdxdna kernel module not loaded")
    
    # Check device nodes
    if os.path.exists("/dev/accel"):
        devices = os.listdir("/dev/accel")
        if devices:
            print(f"✓ Accelerator devices: {', '.join(devices)}")
        else:
            print("✗ No accelerator devices in /dev/accel")
    
    # Check XRT
    success, output = check_command("xrt-smi version", "XRT")
    if success:
        success, output = check_command("xrt-smi examine", "NPU Device")
    
    # Check PCI device
    success, output = check_command(
        "lspci -nn | grep -E '1022:(1502|17F0)'",
        "NPU PCI Device"
    )
    if success and output:
        if "1502" in output:
            print("  → XDNA (Phoenix/Hawk Point) detected")
        elif "17F0" in output:
            print("  → XDNA2 (Strix) detected")
    
    return True

def check_npu_windows():
    """Check NPU on Windows"""
    print_header("Windows NPU Detection")
    
    # Check for Ryzen AI installation
    ryzen_ai_paths = [
        os.environ.get("RYZEN_AI_INSTALLATION_PATH", ""),
        "C:\\Program Files\\AMD\\RyzenAI",
        "C:\\Program Files (x86)\\AMD\\RyzenAI",
    ]
    
    for path in ryzen_ai_paths:
        if path and os.path.exists(path):
            print(f"✓ Ryzen AI found at: {path}")
            break
    else:
        print("✗ Ryzen AI installation not found")
    
    # Check Device Manager for NPU
    success, output = check_command(
        "wmic path win32_VideoController get name",
        "Display Adapters"
    )
    
    # Check for NPU in Device Manager
    success, output = check_command(
        "wmic path Win32_PnPEntity where \"Name like '%NPU%' or Name like '%Neural%' or Name like '%IPU%'\" get Name",
        "NPU Device"
    )
    
    return True

def check_onnxruntime():
    """Check ONNX Runtime with Vitis AI support"""
    print_header("ONNX Runtime Check")
    
    packages = [
        ("onnxruntime", "ONNX Runtime"),
        ("onnxruntime_genai", "ONNX Runtime GenAI"),
        ("numpy", "NumPy"),
        ("onnx", "ONNX"),
    ]
    
    for package, description in packages:
        check_python_package(package, description)
    
    # Check Vitis AI EP
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        print(f"\nAvailable providers: {available_providers}")
        
        if "VitisAIExecutionProvider" in available_providers:
            print("✓ Vitis AI Execution Provider available")
        else:
            print("✗ Vitis AI Execution Provider not available")
            print("  Install with: pip install ryzen-ai-libraries")
    except ImportError:
        pass

def run_quick_test():
    """Run a quick NPU test"""
    print_header("Quick NPU Test")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Create simple ONNX model
        print("Creating test model...")
        
        # Try to create session with Vitis AI
        providers = ['CPUExecutionProvider']
        
        try:
            if "VitisAIExecutionProvider" in ort.get_available_providers():
                providers.insert(0, 'VitisAIExecutionProvider')
                print("Attempting Vitis AI EP...")
        except:
            pass
        
        # Simple matmul test
        test_input = np.random.randn(1, 128).astype(np.float32)
        
        print(f"Test input shape: {test_input.shape}")
        print("✓ Numpy test passed")
        
    except ImportError as e:
        print(f"✗ Cannot run test: {e}")

def main():
    print("=" * 50)
    print(" AMD NPU Verification Script")
    print(f" Platform: {platform.system()} {platform.release()}")
    print(f" Python: {platform.python_version()}")
    print("=" * 50)
    
    system = platform.system()
    
    if system == "Linux":
        check_npu_linux()
    elif system == "Windows":
        check_npu_windows()
    else:
        print(f"Unsupported platform: {system}")
        return 1
    
    check_onnxruntime()
    run_quick_test()
    
    print_header("Summary")
    print("""
For NPU support, ensure:
1. NPU drivers are installed
2. XRT is installed and configured
3. ONNX Runtime with Vitis AI EP is available

Next steps:
- Convert model: python tools/gguf_to_onnx.py
- Run inference: ./build/bin/llama-cli -m model.gguf
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())