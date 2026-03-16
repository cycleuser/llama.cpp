// =============================================================================
// AMD GCN-Specific Tuning for Vulkan Backend
// =============================================================================
// 
// This file contains modifications for AMD GCN/RDNA optimization in Vulkan.
// 
// Installation:
// Copy relevant sections to: ggml/src/ggml-vulkan/ggml-vulkan.cpp
//
// =============================================================================

// -----------------------------------------------------------------------------
// Part 1: Add helper functions at the top
// -----------------------------------------------------------------------------

#define VK_API_VERSION VK_API_VERSION_1_2

// GCN-specific tuning parameters
// GCN architectures (gfx8, gfx9, gfx10) use 64-wide wavefronts
// RDNA3 (gfx11) uses 32-wide wavefronts like NVIDIA

// Check if this is an AMD GCN GPU (not RDNA3)
static bool is_amd_gcn(const vk::PhysicalDeviceProperties& props) {
    if (props.vendorID != 0x1002) return false;  // Not AMD
    
    // RDNA3 (gfx11) uses 32-wide wavefronts
    std::string name = props.deviceName;
    return name.find("gfx11") == std::string::npos;
}

// Get optimal subgroup size for AMD GPUs
static int get_amd_subgroup_size(const vk::PhysicalDeviceProperties& props) {
    // Check environment variable override
    const char* env_size = getenv("GCN_SUBGROUP_SIZE");
    if (env_size) return atoi(env_size);
    
    // Default: 64 for GCN, 32 for RDNA3
    return is_amd_gcn(props) ? 64 : 32;
}

// -----------------------------------------------------------------------------
// Part 2: Add to device initialization (in init_device or similar)
// -----------------------------------------------------------------------------

// AMD GCN-specific tuning
if (device->props.vendorID == 0x1002) {  // AMD vendor ID
    int subgroup_size = get_amd_subgroup_size(device->props);
    
    GGML_VK_LOG_INFO("AMD GPU detected: %s (subgroup size: %d)\n",
        device->props.deviceName.c_str(), subgroup_size);
    
    // Adjust warptile sizes for 64-wide wavefronts
    if (subgroup_size == 64) {
        // GCN architecture - optimize for 64-wide wavefronts
        device->warptile_m = 64;
        device->warptile_n = 64;
        device->warptile_k = 32;
        
        // Smaller batch sizes for better utilization
        device->batch_size = 512;
    } else {
        // RDNA3 - similar to NVIDIA
        device->warptile_m = 32;
        device->warptile_n = 32;
        device->warptile_k = 16;
    }
}

// Flash Attention tuning for GCN
// GCN doesn't have hardware Flash Attention, use optimized fallback
if (is_amd_gcn(device->props)) {
    // Smaller tile sizes for better compute unit utilization
    device->flash_attn_tile_m = 16;
    device->flash_attn_tile_n = 64;
    device->flash_attn_tile_k = 16;
    GGML_VK_LOG_INFO("GCN Flash Attention tuning applied\n");
}

// Conv2D optimization for GCN
// Better compute unit utilization with larger work groups
if (is_amd_gcn(device->props)) {
    device->conv2d_workgroup_x = 16;
    device->conv2d_workgroup_y = 16;
    GGML_VK_LOG_INFO("GCN Conv2D optimization applied\n");
}

// -----------------------------------------------------------------------------
// Part 3: Environment Variables
// -----------------------------------------------------------------------------
// Users can override subgroup size:
//   export GCN_SUBGROUP_SIZE=64
//   export GCN_SUBGROUP_SIZE=32
// -----------------------------------------------------------------------------