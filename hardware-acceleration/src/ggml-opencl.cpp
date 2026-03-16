// =============================================================================
// AMD GPU Support for OpenCL Backend
// =============================================================================
// 
// This file contains modifications to enable AMD GPU support in OpenCL backend.
// 
// Installation:
// Copy relevant sections to: ggml/src/ggml-opencl/ggml-opencl.cpp
//
// =============================================================================

// -----------------------------------------------------------------------------
// Part 1: Add GPU vendor enum (around existing enum)
// -----------------------------------------------------------------------------

enum ggml_opencl_gpu_vendor {
    GGML_OPENCL_GPU_VENDOR_UNKNOWN,
    GGML_OPENCL_GPU_VENDOR_ADRENO,
    GGML_OPENCL_GPU_VENDOR_INTEL,
    GGML_OPENCL_GPU_VENDOR_AMD,      // <-- Add this
    GGML_OPENCL_GPU_VENDOR_NVIDIA,   // <-- Add this
};

// -----------------------------------------------------------------------------
// Part 2: Add wavefront size detection function
// -----------------------------------------------------------------------------

// AMD GPU wavefront sizes
// GCN (gfx8, gfx9, gfx10): 64
// RDNA3 (gfx11): 32 (like NVIDIA warps)
static int get_amd_wavefront_size(const std::string & device_name) {
    if (device_name.find("gfx11") != std::string::npos) return 32;
    return 64;  // Default for GCN architectures
}

// -----------------------------------------------------------------------------
// Part 3: Add to struct ggml_opencl_device
// -----------------------------------------------------------------------------

struct ggml_opencl_device {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
    enum ggml_opencl_gpu_vendor vendor;
    std::string name;
    bool fp16;
    int wavefront_size;  // <-- Add this for AMD GPUs
};

// -----------------------------------------------------------------------------
// Part 4: Modify init_device function to detect AMD GPUs
// -----------------------------------------------------------------------------

static void init_device(ggml_opencl_device * device, cl_device_id cl_device) {
    // ... existing code ...
    
    // Get vendor string
    std::string vendor_str;
    size_t vendor_size;
    clGetDeviceInfo(cl_device, CL_DEVICE_VENDOR, 0, nullptr, &vendor_size);
    vendor_str.resize(vendor_size);
    clGetDeviceInfo(cl_device, CL_DEVICE_VENDOR, vendor_size, &vendor_str[0], nullptr);
    
    // Detect GPU vendor
    if (vendor_str.find("Qualcomm") != std::string::npos ||
        vendor_str.find("Adreno") != std::string::npos) {
        device->vendor = GGML_OPENCL_GPU_VENDOR_ADRENO;
        GGML_OPENCL_LOG_INFO("Adreno GPU detected: %s\n", device->name.c_str());
    } else if (vendor_str.find("Intel") != std::string::npos) {
        device->vendor = GGML_OPENCL_GPU_VENDOR_INTEL;
        GGML_OPENCL_LOG_INFO("Intel GPU detected: %s\n", device->name.c_str());
    } else if (vendor_str.find("AMD") != std::string::npos ||
               vendor_str.find("Advanced Micro Devices") != std::string::npos) {
        device->vendor = GGML_OPENCL_GPU_VENDOR_AMD;
        device->wavefront_size = get_amd_wavefront_size(device->name);
        GGML_OPENCL_LOG_INFO("AMD GPU detected: %s (wavefront: %d)\n", 
            device->name.c_str(), device->wavefront_size);
    } else if (vendor_str.find("NVIDIA") != std::string::npos) {
        device->vendor = GGML_OPENCL_GPU_VENDOR_NVIDIA;
        device->wavefront_size = 32;  // NVIDIA warp size
        GGML_OPENCL_LOG_INFO("NVIDIA GPU detected: %s\n", device->name.c_str());
    }
}