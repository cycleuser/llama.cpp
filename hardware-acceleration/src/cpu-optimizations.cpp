// =============================================================================
// CPU Performance Optimizations
// =============================================================================
//
// This file contains CPU-specific performance optimizations.
//
// Installation:
// Apply relevant patches to: ggml/src/ggml-cpu/ops.cpp
//
// =============================================================================

// -----------------------------------------------------------------------------
// Optimization 1: Improved RMS Norm (replace existing ggml_compute_forward_rms_norm)
// -----------------------------------------------------------------------------

// Original location: ggml/src/ggml-cpu/ops.cpp (around line 3733)
// The current implementation has "// TODO: optimize" comment

/*
static void ggml_compute_forward_rms_norm_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {
    
    const ggml_tensor * src0 = dst->src[0];
    
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    
    const int ith = params->ith;
    const int nth = params->nth;
    
    GGML_TENSOR_UNARY_OP_LOCALS
    
    const float eps = 1e-5f;
    
    // Parallelize over rows
    const int nr = ne11 * ne12 * ne13;
    const int dr = (nr + nth - 1) / nth;
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);
    
    // Process each row
    for (int ir = ir0; ir < ir1; ++ir) {
        // Determine which 3D position this row belongs to
        const int i11 = ir % ne11;
        const int i12 = (ir / ne11) % ne12;
        const int i13 = ir / (ne11 * ne12);
        
        const float * src_row = (const float *) ((char *) src0->data + 
            i11 * nb11 + i12 * nb12 + i13 * nb13);
        float * dst_row = (float *) ((char *) dst->data + 
            i11 * nb1 + i12 * nb2 + i13 * nb3);
        
        // Compute sum of squares using SIMD when available
        float sum = 0.0f;
        
#ifdef __AVX2__
        __m256 sum_vec = _mm256_setzero_ps();
        const int ne0_8 = ne00 & ~7;
        
        for (int i = 0; i < ne0_8; i += 8) {
            __m256 v = _mm256_loadu_ps(src_row + i);
            sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
        }
        
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 sum_128 = _mm_add_ps(hi, lo);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum = _mm_cvtss_f32(sum_128);
        
        // Handle remaining elements
        for (int i = ne0_8; i < ne00; ++i) {
            sum += src_row[i] * src_row[i];
        }
#elif defined(__ARM_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        const int ne0_4 = ne00 & ~3;
        
        for (int i = 0; i < ne0_4; i += 4) {
            float32x4_t v = vld1q_f32(src_row + i);
            sum_vec = vmlaq_f32(sum_vec, v, v);
        }
        
        float sum_array[4];
        vst1q_f32(sum_array, sum_vec);
        sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        for (int i = ne0_4; i < ne00; ++i) {
            sum += src_row[i] * src_row[i];
        }
#else
        for (int i = 0; i < ne00; ++i) {
            sum += src_row[i] * src_row[i];
        }
#endif
        
        const float mean = sum / ne00;
        const float scale = 1.0f / sqrtf(mean + eps);
        
        // Apply normalization with SIMD
#ifdef __AVX2__
        const __m256 scale_vec = _mm256_set1_ps(scale);
        for (int i = 0; i < ne0_8; i += 8) {
            __m256 v = _mm256_loadu_ps(src_row + i);
            _mm256_storeu_ps(dst_row + i, _mm256_mul_ps(v, scale_vec));
        }
        for (int i = ne0_8; i < ne00; ++i) {
            dst_row[i] = src_row[i] * scale;
        }
#elif defined(__ARM_NEON)
        const float32x4_t scale_vec = vdupq_n_f32(scale);
        for (int i = 0; i < ne0_4; i += 4) {
            float32x4_t v = vld1q_f32(src_row + i);
            vst1q_f32(dst_row + i, vmulq_f32(v, scale_vec));
        }
        for (int i = ne0_4; i < ne00; ++i) {
            dst_row[i] = src_row[i] * scale;
        }
#else
        for (int i = 0; i < ne00; ++i) {
            dst_row[i] = src_row[i] * scale;
        }
#endif
    }
}
*/

// -----------------------------------------------------------------------------
// Optimization 2: Improved Group Norm
// -----------------------------------------------------------------------------

// Original location: ggml/src/ggml-cpu/ops.cpp (around line 3802)

/*
static void ggml_compute_forward_group_norm_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {
    
    // Similar SIMD optimizations as RMS norm
    // Group norm is more complex due to grouping
    // Key optimization: use SIMD for sum computation within groups
    
    // ... optimized implementation ...
}
*/

// -----------------------------------------------------------------------------
// Optimization 3: Memory prefetching for large tensors
// -----------------------------------------------------------------------------

/*
// Add to compute functions that iterate over large arrays:

#ifdef __builtin_prefetch
    // Prefetch next cache line while processing current
    if (i + 16 < ne00) {
        __builtin_prefetch(src_row + i + 16, 0, 3);
        __builtin_prefetch(dst_row + i + 16, 1, 3);
    }
#endif
*/

// -----------------------------------------------------------------------------
// Optimization 4: Batch memory allocation (replace multiple mallocs)
// -----------------------------------------------------------------------------

// Original location: src/llama-batch.cpp (lines 889-902)
// Replace individual allocations with single contiguous allocation

/*
struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    struct llama_batch batch;
    batch.n_tokens = 0;
    
    // Calculate total memory needed
    size_t total_size = 0;
    total_size += sizeof(float) * n_tokens_alloc * embd;       // embd
    total_size += sizeof(llama_token) * n_tokens_alloc;        // token
    total_size += sizeof(llama_pos) * n_tokens_alloc;          // pos
    total_size += sizeof(int32_t) * n_tokens_alloc;            // n_seq_id
    total_size += sizeof(llama_seq_id *) * n_tokens_alloc;     // seq_id pointers
    total_size += sizeof(llama_seq_id) * n_tokens_alloc * n_seq_max; // seq_id data
    total_size += sizeof(int8_t) * n_tokens_alloc;             // logits
    
    // Single allocation
    char * base = (char *) malloc(total_size);
    char * ptr = base;
    
    batch.embd = embd ? (float *) ptr : nullptr;
    ptr += sizeof(float) * n_tokens_alloc * embd;
    
    batch.token = (llama_token *) ptr;
    ptr += sizeof(llama_token) * n_tokens_alloc;
    
    batch.pos = (llama_pos *) ptr;
    ptr += sizeof(llama_pos) * n_tokens_alloc;
    
    batch.n_seq_id = (int32_t *) ptr;
    ptr += sizeof(int32_t) * n_tokens_alloc;
    
    batch.seq_id = (llama_seq_id **) ptr;
    ptr += sizeof(llama_seq_id *) * n_tokens_alloc;
    
    // Point seq_id entries to contiguous data
    llama_seq_id * seq_data = (llama_seq_id *) ptr;
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = seq_data + i * n_seq_max;
        batch.n_seq_id[i] = 0;
    }
    
    batch.logits = (int8_t *)(seq_data + n_tokens_alloc * n_seq_max);
    
    // Store base pointer for freeing
    batch._base = base;  // New field needed in struct
    
    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch._base) {
        free(batch._base);
    }
}
*/

// -----------------------------------------------------------------------------
// Optimization 5: Thread pool warmup
// -----------------------------------------------------------------------------

/*
// Add to ggml-cpu.c initialization to reduce thread startup overhead

static void ggml_cpu_thread_pool_init(struct ggml_cpu_thread_pool * pool, int n_threads) {
    pool->n_threads = n_threads;
    pool->threads = (pthread_t *) malloc(sizeof(pthread_t) * n_threads);
    
    // Create threads in advance and keep them waiting on a condition variable
    // This reduces latency for the first parallel operation
    
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);
    
    for (int i = 0; i < n_threads; ++i) {
        pthread_create(&pool->threads[i], NULL, ggml_cpu_thread_func, pool);
    }
}
*/