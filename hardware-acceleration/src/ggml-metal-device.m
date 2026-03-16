// =============================================================================
// Apple M-Series Chip Detection for Metal Backend
// =============================================================================
// 
// This file contains modifications to detect and report Apple Silicon chip details.
// 
// Installation:
// 1. Copy relevant sections to the corresponding files
// 2. Or use as reference for manual integration
//
// Files to modify:
// - ggml/src/ggml-metal/ggml-metal-impl.h
// - ggml/src/ggml-metal/ggml-metal-device.h  
// - ggml/src/ggml-metal/ggml-metal-device.m
//
// =============================================================================

// -----------------------------------------------------------------------------
// Part 1: Add to ggml-metal-impl.h (at the beginning)
// -----------------------------------------------------------------------------

// Apple Silicon chip generations
enum ggml_metal_chip_generation {
    GGML_METAL_CHIP_GENERATION_UNKNOWN = 0,
    GGML_METAL_CHIP_GENERATION_M1      = 1,  // M1, M1 Pro, M1 Max, M1 Ultra
    GGML_METAL_CHIP_GENERATION_M2      = 2,  // M2, M2 Pro, M2 Max, M2 Ultra
    GGML_METAL_CHIP_GENERATION_M3      = 3,  // M3, M3 Pro, M3 Max
    GGML_METAL_CHIP_GENERATION_M4      = 4,  // M4, M4 Pro, M4 Max
};

// Chip-specific parameters
struct ggml_metal_chip_params {
    int gpu_cores;
    int memory_bandwidth_gbps;  // GB/s
    int recommended_threads;
};

// M-series chip parameters (approximate values)
// Base models:
// M1:     7-8 cores,  100 GB/s
// M2:     10 cores,  100 GB/s  
// M3:     10 cores,  100 GB/s
// M4:     10 cores,  120 GB/s

// -----------------------------------------------------------------------------
// Part 2: Add to ggml-metal-device.h (in struct ggml_metal_device)
// -----------------------------------------------------------------------------

struct ggml_metal_device {
    // ... existing fields ...
    
    // Add these new fields:
    enum ggml_metal_chip_generation chip_generation;
    int gpu_cores;
    int memory_bandwidth_gbps;
};

// -----------------------------------------------------------------------------
// Part 3: Add to ggml-metal-device.m (in initialization function)
// -----------------------------------------------------------------------------

// Add this function before ggml_metal_device_init:

static void detect_apple_silicon(struct ggml_metal_device * device, id<MTLDevice> mtl_device) {
    NSString * deviceName = [mtl_device name];
    const char * name = [deviceName UTF8String];
    
    // Detect M-series chip generation
    if (strstr(name, "M4") != NULL) {
        device->chip_generation = GGML_METAL_CHIP_GENERATION_M4;
        device->gpu_cores = 10;  // Base M4
        device->memory_bandwidth_gbps = 120;
    } else if (strstr(name, "M3") != NULL) {
        device->chip_generation = GGML_METAL_CHIP_GENERATION_M3;
        device->gpu_cores = 10;
        device->memory_bandwidth_gbps = 100;
    } else if (strstr(name, "M2") != NULL) {
        device->chip_generation = GGML_METAL_CHIP_GENERATION_M2;
        device->gpu_cores = 10;
        device->memory_bandwidth_gbps = 100;
    } else if (strstr(name, "M1") != NULL) {
        device->chip_generation = GGML_METAL_CHIP_GENERATION_M1;
        device->gpu_cores = 7;  // Base M1
        device->memory_bandwidth_gbps = 100;
    } else {
        device->chip_generation = GGML_METAL_CHIP_GENERATION_UNKNOWN;
        device->gpu_cores = 0;
        device->memory_bandwidth_gbps = 0;
    }
    
    // Detect Pro/Max variants
    if (strstr(name, "Max") != NULL) {
        device->gpu_cores *= 3;  // Max has ~3x cores
        device->memory_bandwidth_gbps *= 3;
    } else if (strstr(name, "Pro") != NULL) {
        device->gpu_cores *= 2;  // Pro has ~2x cores
        device->memory_bandwidth_gbps *= 2;
    }
}

// Then call this function in ggml_metal_device_init, after getting the device:
// detect_apple_silicon(device, mtl_device);

// Add logging:
// GGML_METAL_LOG_INFO("Apple %s series detected (%d GPU cores, %d GB/s)\n",
//     name, device->gpu_cores, device->memory_bandwidth_gbps);

// -----------------------------------------------------------------------------
// Example output:
// -----------------------------------------------------------------------------
// ggml_metal_device_init: Apple M4 series detected (10 GPU cores, 120 GB/s)
// ggml_metal_device_init: Apple M2 Pro series detected (19 GPU cores, 200 GB/s)
// -----------------------------------------------------------------------------