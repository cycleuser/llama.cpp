/**
 * AMD XDNA NPU Backend for GGML
 * 
 * This backend provides hardware acceleration for AMD Ryzen AI NPU
 * using the Xilinx Runtime (XRT) and Vitis AI Execution Provider.
 * 
 * Supported Hardware:
 * - XDNA (Phoenix/Hawk Point): Ryzen 7040/8040 series
 * - XDNA2 (Strix): Ryzen AI 300 series
 */

#ifndef GGML_AMDXDNA_H
#define GGML_AMDXDNA_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMD XDNA device types
typedef enum {
    GGML_AMDXDNA_TYPE_XDNA,     // Phoenix/Hawk Point (Gen 1)
    GGML_AMDXDNA_TYPE_XDNA2,    // Strix (Gen 2)
    GGML_AMDXDNA_TYPE_UNKNOWN,
} ggml_amdxdna_device_type_t;

// AMD XDNA device info
typedef struct {
    int device_id;
    char name[256];
    ggml_amdxdna_device_type_t type;
    size_t memory_size;         // NPU local memory
    size_t compute_units;       // AIE-ML tiles
    int tops;                   // TOPS performance
    bool supports_bf16;
    bool supports_int8;
} ggml_amdxdna_device_info_t;

// AMD XDNA context
typedef struct ggml_backend_amdxdna_context ggml_backend_amdxdna_context_t;

// Initialization and cleanup
GGML_API bool ggml_amdxdna_is_available(void);
GGML_API int ggml_amdxdna_get_device_count(void);
GGML_API bool ggml_amdxdna_get_device_info(int device, ggml_amdxdna_device_info_t * info);

// Backend creation
GGML_API ggml_backend_t ggml_backend_amdxdna_init(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_amdxdna_buffer_type(int device);

// Buffer management
GGML_API ggml_backend_buffer_t ggml_backend_amdxdna_buffer_from_ptr(
    void * ptr,
    size_t size,
    size_t max_tensor_size
);

// Context management
GGML_API ggml_backend_amdxdna_context_t * ggml_amdxdna_context_create(int device);
GGML_API void ggml_amdxdna_context_free(ggml_backend_amdxdna_context_t * ctx);

// Model compilation (for XDNA)
GGML_API bool ggml_amdxdna_compile_model(
    ggml_backend_amdxdna_context_t * ctx,
    struct ggml_cgraph * graph,
    const char * cache_path
);

// Hybrid execution support
GGML_API bool ggml_amdxdna_supports_hybrid(void);
GGML_API void ggml_amdxdna_set_hybrid_mode(bool enabled);

// Performance tuning
GGML_API void ggml_amdxdna_set_num_threads(int n_threads);
GGML_API void ggml_amdxdna_set_memory_limit(size_t limit);

// XRT specific functions
GGML_API bool ggml_amdxdna_xrt_init(void);
GGML_API void ggml_amdxdna_xrt_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_AMDXDNA_H