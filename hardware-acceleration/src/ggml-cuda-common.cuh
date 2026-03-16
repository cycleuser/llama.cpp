// =============================================================================
// AMD GCN4 (gfx803: RX580/RX590/Polaris) dp4a Optimization
// =============================================================================
// 
// This file contains modifications to enable dp4a optimization for AMD GCN4.
// 
// Installation:
// 1. Copy this file to: ggml/src/ggml-cuda/common.cuh
// 2. Or apply the specific changes marked below to your existing file
//
// Changes from original:
// - Added __gfx803__ (GCN4) to RDNA3/RDNA4 dp4a path
// - Added __gfx803__ to V_DOT2_F32_F16_AVAILABLE macro
//
// Technical Details:
// - GCN4 (gfx803) includes: RX580, RX590, RX570, RX560, Polaris series
// - Uses __builtin_amdgcn_sudot4 intrinsic (same as RDNA3/RDNA4)
// - GCN4 has 64-wide wavefronts (vs NVIDIA's 32-wide warps)
// - Has fast FP16, no tensor cores, no Flash Attention hardware
//
// =============================================================================

// -----------------------------------------------------------------------------
// CHANGE 1: Modify dp4a function (around line 670)
// -----------------------------------------------------------------------------
// 
// ORIGINAL CODE:
//   #elif defined(RDNA3) || defined(RDNA4)
//       c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
//
// MODIFIED CODE:
//   #elif defined(RDNA3) || defined(RDNA4) || defined(__gfx803__)
//       c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
//
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// CHANGE 2: Modify V_DOT2_F32_F16_AVAILABLE macro (around line 715)
// -----------------------------------------------------------------------------
//
// ORIGINAL CODE:
//   #if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
//   #define V_DOT2_F32_F16_AVAILABLE
//   #endif
//
// MODIFIED CODE:
//   #if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx803__) || defined(__gfx906__) || defined(CDNA))
//   #define V_DOT2_F32_F16_AVAILABLE
//   #endif
//
// -----------------------------------------------------------------------------

// Complete modified function:

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4) || defined(__gfx803__)  // <-- Added __gfx803__
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else // defined(GGML_USE_HIP)

#if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)

#endif // defined(GGML_USE_HIP)
}

// V_DOT2_F32_F16_AVAILABLE macro modification:
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx803__) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif