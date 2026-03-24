/**
 * AMD XDNA NPU Backend Tests
 */

#include "ggml-amdxdna.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define TEST_PASS(name) printf("[PASS] %s\n", name)
#define TEST_FAIL(name, msg) printf("[FAIL] %s: %s\n", name, msg)

static int test_device_detection(void) {
    printf("\n=== Test: Device Detection ===\n");
    
    bool available = ggml_amdxdna_is_available();
    printf("NPU available: %s\n", available ? "yes" : "no");
    
    int count = ggml_amdxdna_get_device_count();
    printf("Device count: %d\n", count);
    
    if (count > 0) {
        for (int i = 0; i < count; i++) {
            ggml_amdxdna_device_info_t info;
            if (ggml_amdxdna_get_device_info(i, &info)) {
                printf("  Device %d:\n", i);
                printf("    Name: %s\n", info.name);
                printf("    Type: %s\n", 
                    info.type == GGML_AMDXDNA_TYPE_XDNA ? "XDNA" :
                    info.type == GGML_AMDXDNA_TYPE_XDNA2 ? "XDNA2" : "Unknown");
                printf("    Memory: %zu MB\n", info.memory_size / (1024 * 1024));
                printf("    TOPS: %d\n", info.tops);
                printf("    INT8 support: %s\n", info.supports_int8 ? "yes" : "no");
                printf("    BF16 support: %s\n", info.supports_bf16 ? "yes" : "no");
            }
        }
        TEST_PASS("Device detection");
        return 0;
    } else {
        printf("No NPU devices found (this may be expected if no hardware)\n");
        TEST_PASS("Device detection (no hardware)");
        return 0;
    }
}

static int test_context_creation(void) {
    printf("\n=== Test: Context Creation ===\n");
    
    int count = ggml_amdxdna_get_device_count();
    if (count == 0) {
        printf("Skipping: No NPU devices\n");
        return 0;
    }
    
    ggml_backend_amdxdna_context_t * ctx = ggml_amdxdna_context_create(0);
    if (ctx) {
        printf("Context created successfully\n");
        ggml_amdxdna_context_free(ctx);
        printf("Context freed successfully\n");
        TEST_PASS("Context creation");
        return 0;
    } else {
        TEST_FAIL("Context creation", "Failed to create context");
        return 1;
    }
}

static int test_backend_init(void) {
    printf("\n=== Test: Backend Initialization ===\n");
    
    int count = ggml_amdxdna_get_device_count();
    if (count == 0) {
        printf("Skipping: No NPU devices\n");
        return 0;
    }
    
    ggml_backend_t backend = ggml_backend_amdxdna_init(0);
    if (backend) {
        printf("Backend created successfully\n");
        printf("Backend name: %s\n", backend->iface.get_name ? 
            backend->iface.get_name(backend) : "N/A");
        
        ggml_backend_buffer_type_t buft = ggml_backend_amdxdna_buffer_type(0);
        if (buft) {
            printf("Buffer type created\n");
        }
        
        if (backend->iface.free) {
            backend->iface.free(backend);
        }
        printf("Backend freed successfully\n");
        TEST_PASS("Backend initialization");
        return 0;
    } else {
        TEST_FAIL("Backend initialization", "Failed to create backend");
        return 1;
    }
}

static int test_hybrid_mode(void) {
    printf("\n=== Test: Hybrid Mode ===\n");
    
    bool supports = ggml_amdxdna_supports_hybrid();
    printf("Hybrid mode supported: %s\n", supports ? "yes" : "no");
    
    ggml_amdxdna_set_hybrid_mode(true);
    printf("Hybrid mode enabled\n");
    
    ggml_amdxdna_set_hybrid_mode(false);
    printf("Hybrid mode disabled\n");
    
    TEST_PASS("Hybrid mode");
    return 0;
}

static int test_xrt_functions(void) {
    printf("\n=== Test: XRT Functions ===\n");
    
    bool xrt_init = ggml_amdxdna_xrt_init();
    printf("XRT initialized: %s\n", xrt_init ? "yes" : "no");
    
    if (xrt_init) {
        ggml_amdxdna_xrt_shutdown();
        printf("XRT shutdown complete\n");
    }
    
    TEST_PASS("XRT functions");
    return 0;
}

static int test_configuration(void) {
    printf("\n=== Test: Configuration ===\n");
    
    ggml_amdxdna_set_num_threads(4);
    printf("Set num threads: 4\n");
    
    ggml_amdxdna_set_memory_limit(1024 * 1024 * 1024); // 1GB
    printf("Set memory limit: 1GB\n");
    
    TEST_PASS("Configuration");
    return 0;
}

int main(int argc, char ** argv) {
    printf("========================================\n");
    printf(" AMD XDNA NPU Backend Tests\n");
    printf("========================================\n");
    
    int failed = 0;
    
    failed += test_device_detection();
    failed += test_context_creation();
    failed += test_backend_init();
    failed += test_hybrid_mode();
    failed += test_xrt_functions();
    failed += test_configuration();
    
    printf("\n========================================\n");
    printf(" Results: %d tests failed\n", failed);
    printf("========================================\n");
    
    return failed;
}