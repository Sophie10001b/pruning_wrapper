#pragma once
#include "utils.cuh"

#define FILLZERO flashinfer::cp_async::SharedMemFillMode::kFillZero;
#define NOFILL flashinfer::cp_async::SharedMemFillMode::kNoFill;

#define PREFETCH flashinfer::cp_async::PrefetchMode::kPrefetch;
#define NOPREFETCH flashinfer::cp_async::PrefetchMode::kNoPrefetch;

#define INITACC flashinfer::mma::MMAMode::kInit;
#define UPDATEACC flashinfer::mma::MMAMode::kInplaceUpdate;

template<typename T>
struct PackedHalf2;

// fp16特化
template<>
struct PackedHalf2<half> {
    using Type = __half2;
    static __device__ __forceinline__ Type pack(float* a) {
        return __floats2half2_rn(*a, *(a + 1));
    }
};

// bf16特化  
template<>
struct PackedHalf2<__nv_bfloat16> {
    using Type = __nv_bfloat162;
    static __device__ __forceinline__ Type pack(float* a) {
        return __floats2bfloat162_rn(*a, *(a + 1));
    }
};


template <typename T, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t num_stages>
struct SharedMem {
    alignas(16) T x_smem[num_stages][BLOCK_M][BLOCK_K];
    alignas(16) T w_smem[num_stages][BLOCK_K][BLOCK_N];
};

// block loop across BK with 128 threads / 4 warps, i.e., consume 16x64 matrices in each mma wave
template <
    typename dot_dtype_, typename acc_dtype_, typename idx_dtype_,
    uint32_t M_, uint32_t N_, uint32_t K_, uint32_t G_, uint32_t NG_,
    uint32_t BLOCK_M_, uint32_t BLOCK_N_, uint32_t BLOCK_K_,
    uint32_t vec_size_, uint32_t num_stages_
>
struct KernelTraits {
    static constexpr uint32_t num_stages = num_stages_;
    static constexpr uint32_t num_warps = 4;
    static constexpr uint32_t num_threads = 4 * NUM_THREADS_PER_WARP;
    static constexpr uint32_t vec_size = vec_size_;

    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;
    static constexpr uint32_t G = G_;
    static constexpr uint32_t NG = NG_;
    static constexpr uint32_t BLOCK_M = BLOCK_M_;
    static constexpr uint32_t BLOCK_N = BLOCK_N_;
    static constexpr uint32_t BLOCK_K = BLOCK_K_;
    static_assert((BLOCK_M / 16) * (BLOCK_N / 16) >= num_warps);
    static_assert(BLOCK_K % 64 == 0);

    using dot_dtype = dot_dtype_;
    using acc_dtype = acc_dtype_;
    using idx_dtype = idx_dtype_;
    using gmem_st_pack_dtype = typename PackedHalf2<dot_dtype>::Type;

    using smem = SharedMem<dot_dtype, BLOCK_M, BLOCK_N, BLOCK_K, num_stages>;
    static constexpr flashinfer::SwizzleMode swizzle_x_smem = flashinfer::SwizzleMode::k128B;
    static constexpr flashinfer::SwizzleMode swizzle_w_smem = flashinfer::SwizzleMode::k128B;

    static constexpr uint32_t load_threads_per_col = BLOCK_K / vec_size;
    static constexpr uint32_t load_row_per_wave = num_threads / load_threads_per_col;
    static constexpr uint32_t bm_load_wave_num = BLOCK_M / load_row_per_wave;
    static constexpr uint32_t bn_load_wave_num = BLOCK_N / load_row_per_wave;

    static constexpr uint32_t mma_num_bm = BLOCK_M / 16;
    static constexpr uint32_t mma_num_bn = BLOCK_N / 16;
    static constexpr uint32_t mma_num_bk = BLOCK_K / 16;
};

template <typename data_dtype, typename idx_dtype>
struct KernelParams {
    data_dtype * __restrict__ x_ptr;
    data_dtype * __restrict__ w_ptr;
    data_dtype * __restrict__ b_ptr;
    data_dtype * __restrict__ o_ptr;
    uint8_t * __restrict__ route_mask_ptr;
    idx_dtype * __restrict__ route_indices_ptr;
};

template <uint32_t num_stages>
__device__ __forceinline__ uint32_t get_stage(const uint32_t stage_id) {
    return (stage_id >= num_stages) ? stage_id - num_stages : stage_id;
}

template <typename KernelTraits, typename KernelParams>
__global__ __launch_bounds__(KernelTraits::num_threads) void BNSparseMLPKernel(KernelParams* params) {
    using dot_dtype = typename KernelTraits::dot_dtype;
    using acc_dtype = typename KernelTraits::acc_dtype;
    using idx_dtype = typename KernelTraits::idx_dtype;
    using gmem_st_pack_dtype = typename KernelTraits::gmem_st_pack_dtype;

    constexpr uint32_t num_stages = KernelTraits::num_stages;
    constexpr uint32_t vec_size = KernelTraits::vec_size;

    constexpr uint32_t M = KernelTraits::M;
    constexpr uint32_t N = KernelTraits::N;
    constexpr uint32_t K = KernelTraits::K;
    constexpr uint32_t G = KernelTraits::G;
    constexpr uint32_t NG = KernelTraits::NG;
    constexpr uint32_t BLOCK_M = KernelTraits::BLOCK_M;
    constexpr uint32_t BLOCK_N = KernelTraits::BLOCK_N;
    constexpr uint32_t BLOCK_K = KernelTraits::BLOCK_K;

    // shared memory settings
    constexpr flashinfer::SwizzleMode swizzle_x_smem = KernelTraits::swizzle_x_smem;
    constexpr flashinfer::SwizzleMode swizzle_w_smem = KernelTraits::swizzle_w_smem;
    extern __shared__ uint8_t smem_storage[];
    auto& smem = reinterpret_cast<typename KernelTraits::smem&>(smem_storage);
    flashinfer::smem_t<swizzle_x_smem> x_smem(smem.x_smem);
    flashinfer::smem_t<swizzle_w_smem> w_smem(smem.w_smem);

    // x每[1, BK] loading需要的threads num
    constexpr uint32_t load_threads_per_col = KernelTraits::load_threads_per_col;
    // 整个CTA单次load m 行数
    constexpr uint32_t load_row_per_wave = KernelTraits::load_row_per_wave;
    // 完整load[BM, BK]需要的循环次数，每次循环load load_row_per_wave = (num_threads / load_threads_per_col) 行
    constexpr uint32_t bm_load_wave_num = KernelTraits::bm_load_wave_num;
    constexpr uint32_t bn_load_wave_num = KernelTraits::bn_load_wave_num;

    constexpr uint32_t mma_num_bm = KernelTraits::mma_num_bm;
    constexpr uint32_t mma_num_bn = KernelTraits::mma_num_bn;
    constexpr uint32_t mma_num_bk = KernelTraits::mma_num_bk;

    const uint32_t tid = threadIdx.x;
    const uint32_t mid = blockIdx.x;
    const uint32_t nid = blockIdx.y;
    const uint32_t num_threads = KernelTraits::num_threads;
    const uint32_t num_warps = KernelTraits::num_warps;
    const uint32_t warp_id = tid / NUM_THREADS_PER_WARP;

    // cooperative group settings for warp/block reduce
    cg::thread_block block = cg::this_thread_block();
    auto block_tile = cg::tiled_partition<num_threads>(block);
    auto warp_tile = cg::tiled_partition<NUM_THREADS_PER_WARP>(block);
    uint32_t lane_id = warp_tile.thread_rank();

    dot_dtype* x_ptr = params->x_ptr;
    dot_dtype* w_ptr = params->w_ptr;
    dot_dtype* b_ptr = params->b_ptr;
    dot_dtype* o_ptr = params->o_ptr;
    uint8_t* route_mask_ptr = params->route_mask_ptr;
    idx_dtype* route_indices_ptr = params->route_indices_ptr;

    // calculate base offset
    const uint32_t base_m_offset = mid * BLOCK_M;
    const uint32_t base_n_offset = nid * BLOCK_N;
    const uint32_t route_idx_offset = nid * M + mid * BLOCK_M;

    // 查询route_mask中每个thread对应m row是否激活
    uint8_t load_valid_mask[bm_load_wave_num];
    uint32_t load_valid_indices[bm_load_wave_num];
    uint8_t load_valid_mask_warp[bm_load_wave_num];

    uint32_t mma_valid_indices[mma_num_bm * 2];
    uint8_t mma_valid_mask_warp[mma_num_bm];

    #pragma unroll
    for (uint32_t i=0; i < bm_load_wave_num; ++i) {
        if (mid * BLOCK_M + i * load_row_per_wave < M) {
            load_valid_mask[i] = route_mask_ptr[route_idx_offset + (tid / load_threads_per_col) + i * load_row_per_wave];
            load_valid_indices[i] = route_indices_ptr[route_idx_offset + (tid / load_threads_per_col) + i * load_row_per_wave];
        }
        else {
            load_valid_mask[i] = 0;
            load_valid_indices[i] = 0;
        }
        // skip smem/reg loading and mma if all m in warp are inactive
        load_valid_mask_warp[i] = cg::reduce(warp_tile, load_valid_mask[i], cg::bit_or<uint8_t>());
    }

    // if all bm are inactive, skip the block
    uint8_t bm_active = 0;
    # pragma unroll
    for (uint32_t i=0; i < bm_load_wave_num; ++i) bm_active |= load_valid_mask_warp[i];
    auto block_active = cg::reduce(block_tile, bm_active, cg::bit_or<uint8_t>());
    if (!block_active) return;

    // calculate whether each mma tile need to be computed
    #pragma unroll
    for (uint32_t i=0; i < mma_num_bm; ++i) {
        uint8_t mma_active = 0;
        #pragma unroll
        for (uint32_t j=0; j < bm_load_wave_num / mma_num_bm; ++j) {
            mma_active |= load_valid_mask_warp[i * (bm_load_wave_num / mma_num_bm) + j];
        }
        mma_valid_mask_warp[i] = mma_active;
        // load indices for writing mma results
        const uint32_t mma_route_idx_offset = route_idx_offset + i * 16 + lane_id / 4;
        mma_valid_indices[i * 2] = (mma_route_idx_offset < M && route_mask_ptr[mma_route_idx_offset]) ? route_indices_ptr[mma_route_idx_offset] : -1;
        mma_valid_indices[i * 2 + 1] = (mma_route_idx_offset < M && route_mask_ptr[mma_route_idx_offset + 8]) ? route_indices_ptr[mma_route_idx_offset + 8] : -1;
    }

    constexpr uint32_t k_tile_num = CDIV(K, BLOCK_K);
    uint32_t k_tile_id = 0;
    uint32_t load_stage = 0;
    uint32_t compute_stage = 0;
    
    // load x[BM, BK]
    uint32_t bk_offset_gmem = (tid % load_threads_per_col) * vec_size;
    uint32_t bm_offset_smem = (tid / load_threads_per_col) * BLOCK_K;
    uint32_t bk_offset_smem = bk_offset_gmem;
    #pragma unroll
    for (uint32_t i=0; i < bm_load_wave_num; ++i){
        uint32_t bm_offset_gmem = load_valid_indices[i] * K;
        if (load_valid_mask_warp[i]) {
            x_smem.load_128b_async<NOFILL>(
                bm_offset_smem + bk_offset_smem + i * load_row_per_wave,
                x_ptr + bm_offset_gmem + bk_offset_gmem + i * load_row_per_wave
            );
        }
    }
    // load w[BLOCK_N, BLOCK_K]
    uint32_t bn_offset_gmem = nid * G * K;
    uint32_t bn_offset_smem = bm_offset_smem;
    #pragma unroll
    for (uint32_t i=0; i < bn_load_wave_num; ++i){
        w_smem.load_128b_async<NOFILL>(
            bn_offset_smem + bk_offset_smem + i * load_row_per_wave,
            w_ptr + bn_offset_gmem + bk_offset_gmem + i * load_row_per_wave
        );
    }
    flashinfer::cp_async::commit_group();
    flashinfer::cp_async::wait_group<num_stages - 1>();
    __syncthreads();

    k_tile_id++;
    uint32_t x_reg[mma_num_bm][mma_num_bk][4];
    uint32_t w_reg[mma_num_bn][mma_num_bk][4];
    acc_dtype acc_reg[mma_num_bm][mma_num_bn][8];

    constexpr uint32_t m_per_warp = BLOCK_M / num_warps;

    // mma m16n16k16
    for (; k_tile_id < k_tile_num; ++k_tile_id) {
        // load next buffer
        load_stage = get_stage<num_stages>(load_stage + 1);
        // load x[BLOCK_M, BLOCK_K]
        bk_offset_gmem += BLOCK_K;
        #pragma unroll
        for (uint32_t i=0; i < bm_load_wave_num; ++i){
            uint32_t bm_offset_gmem = load_valid_indices[i] * K;
            if (load_valid_mask_warp[i]) {
                x_smem.load_128b_async<NOFILL>(
                    bm_offset_smem + bk_offset_smem + i * load_row_per_wave + load_stage * BLOCK_M * BLOCK_K,
                    x_ptr + bm_offset_gmem + bk_offset_gmem + i * load_row_per_wave
                );
            }
        }
        // load w[BLOCK_N, BLOCK_K]
        #pragma unroll
        for (uint32_t i=0; i < bn_load_wave_num; ++i){
            w_smem.load_128b_async<NOFILL>(
                bn_offset_smem + bk_offset_smem + i * load_row_per_wave + load_stage * BLOCK_N * BLOCK_K,
                w_ptr + bn_offset_gmem + bk_offset_gmem + i * load_row_per_wave
            );
        }

        #pragma unroll
        for (uint32_t mma_id_bk=0; mma_id_bk < mma_num_bk; ++mma_id_bk) {
            // load x[BLOCK_M, BLOCK_K] to reg
            #pragma unroll
            for (uint32_t mma_id_bm=0; mma_id_bm < mma_num_bm; ++mma_id_bm) {
                if (!mma_valid_mask_warp[mma_id_bm]) continue;
                const uint32_t row = mma_id_bm * 16 + warp_id * m_per_warp + (lane_id % 16);
                const uint32_t col = mma_id_bk * 16 + (lane_id / 16 * 8);
                const uint32_t offset = (compute_stage * BLOCK_M * BLOCK_K + row * BLOCK_K + col) * sizeof(dot_dtype);
                flashinfer::mma::ldmatrix_m8n8x4(x_reg[mma_id_bm][mma_id_bk], x_smem + offset);
            }
            // load w[BLOCK_K, BLOCK_N] (transposed) to reg
            #pragma unroll
            for (uint32_t mma_id_bn=0; mma_id_bn < mma_num_bn; ++mma_id_bn) {
                const uint32_t row = mma_id_bn * 16 + warp_id * m_per_warp + (lane_id % 16);
                const uint32_t col = mma_id_bk * 16 + (lane_id / 16 * 8);
                const uint32_t offset = (compute_stage * BLOCK_N * BLOCK_K + row * BLOCK_K + col) * sizeof(dot_dtype);
                flashinfer::mma::ldmatrix_m8n8x4_trans(w_reg[mma_id_bn][mma_id_bk], w_smem + offset);
            }
            // mma compute
            #pragma unroll
            for (uint32_t mma_id_bm=0; mma_id_bm < mma_num_bm; ++mma_id_bm) {
                if (!mma_valid_mask_warp[mma_id_bm]) continue;
                #pragma unroll
                for (uint32_t mma_id_bn=0; mma_id_bn < mma_num_bn; ++mma_id_bn) {
                    if (mma_id_bk == 0) {
                        flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KernelTraits::dot_dtype, INITACC>(
                            acc_reg[mma_id_bm][mma_id_bn],
                            x_reg[mma_id_bm][mma_id_bk],
                            w_reg[mma_id_bn][mma_id_bk]
                        );
                    }
                    else {
                        flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KernelTraits::dot_dtype, UPDATEACC>(
                            acc_reg[mma_id_bm][mma_id_bn],
                            x_reg[mma_id_bm][mma_id_bk],
                            w_reg[mma_id_bn][mma_id_bk]
                        );
                    }
                }
            }
        }
        compute_stage = get_stage<num_stages>(compute_stage + 1);
        flashinfer::cp_async::commit_group();
        flashinfer::cp_async::wait_group<num_stages - 1>();
        __syncthreads();
    }

    // compute last stage
    #pragma unroll
    for (uint32_t mma_id_bk=0; mma_id_bk < mma_num_bk; ++mma_id_bk) {
        // load x[BLOCK_M, BLOCK_K] to reg
        #pragma unroll
        for (uint32_t mma_id_bm=0; mma_id_bm < mma_num_bm; ++mma_id_bm) {
            if (!mma_valid_mask_warp[mma_id_bm]) continue;
            const uint32_t row = mma_id_bm * 16 + warp_id * m_per_warp + (lane_id % 16);
            const uint32_t col = mma_id_bk * 16 + (lane_id / 16 * 8);
            const uint32_t offset = (compute_stage * BLOCK_M * BLOCK_K + row * BLOCK_K + col) * sizeof(dot_dtype);
            flashinfer::mma::ldmatrix_m8n8x4(x_reg[mma_id_bm][mma_id_bk], x_smem + offset);
        }
        // load w[BLOCK_K, BLOCK_N] (transposed) to reg
        #pragma unroll
        for (uint32_t mma_id_bn=0; mma_id_bn < mma_num_bn; ++mma_id_bn) {
            const uint32_t row = mma_id_bn * 16 + warp_id * m_per_warp + (lane_id % 16);
            const uint32_t col = mma_id_bk * 16 + (lane_id / 16 * 8);
            const uint32_t offset = (compute_stage * BLOCK_N * BLOCK_K + row * BLOCK_K + col) * sizeof(dot_dtype);
            flashinfer::mma::ldmatrix_m8n8x4_trans(w_reg[mma_id_bn][mma_id_bk], w_smem + offset);
        }
        // mma compute
        #pragma unroll
        for (uint32_t mma_id_bm=0; mma_id_bm < mma_num_bm; ++mma_id_bm) {
            if (!mma_valid_mask_warp[mma_id_bm]) continue;
            #pragma unroll
            for (uint32_t mma_id_bn=0; mma_id_bn < mma_num_bn; ++mma_id_bn) {
                if (mma_id_bk == 0) {
                    flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KernelTraits::dot_dtype, INITACC>(
                        acc_reg[mma_id_bm][mma_id_bn],
                        x_reg[mma_id_bm][mma_id_bk],
                        w_reg[mma_id_bn][mma_id_bk]
                    );
                }
                else {
                    flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KernelTraits::dot_dtype, UPDATEACC>(
                        acc_reg[mma_id_bm][mma_id_bn],
                        x_reg[mma_id_bm][mma_id_bk],
                        w_reg[mma_id_bn][mma_id_bk]
                    );
                }
            }
        }
    }

    // store to gmem
    const uint32_t gmem_st_n_offset = nid * BLOCK_N + (lane_id % 4) * 2;
    #pragma unroll
    for (uint32_t mma_id_bm=0; mma_id_bm < mma_num_bm; ++mma_id_bm) {
        if (!mma_valid_mask_warp[mma_id_bm]) continue;
        const uint32_t block_0_valid = mma_valid_indices[mma_id_bm * 2] > 0;
        const uint32_t block_1_valid = mma_valid_indices[mma_id_bm * 2 + 1] > 0;

        #pragma unroll
        for (uint32_t mma_id_bn=0; mma_id_bn < mma_num_bn; ++mma_id_bn) {
            if (block_0_valid) {
                *reinterpret_cast<gmem_st_pack_dtype*>(o_ptr + mma_valid_indices[mma_id_bm * 2] * N + mma_id_bn * 16 + gmem_st_n_offset) = PackedHalf2<dot_dtype>::pack(&acc_reg[mma_id_bm][mma_id_bn][0]);
                *reinterpret_cast<gmem_st_pack_dtype*>(o_ptr + mma_valid_indices[mma_id_bm * 2] * N + mma_id_bn * 16 + gmem_st_n_offset + 8) = PackedHalf2<dot_dtype>::pack(&acc_reg[mma_id_bm][mma_id_bn][4]);
            }
            if (block_1_valid) {
                *reinterpret_cast<gmem_st_pack_dtype*>(o_ptr + mma_valid_indices[mma_id_bm * 2 + 1] * N + mma_id_bn * 16 + gmem_st_n_offset) = PackedHalf2<dot_dtype>::pack(&acc_reg[mma_id_bm][mma_id_bn][2]);
                *reinterpret_cast<gmem_st_pack_dtype*>(o_ptr + mma_valid_indices[mma_id_bm * 2 + 1] * N + mma_id_bn * 16 + gmem_st_n_offset + 8) = PackedHalf2<dot_dtype>::pack(&acc_reg[mma_id_bm][mma_id_bn][6]);
            }
        }
    }
}

template <typename KernelTraits, typename KernelParams>
cudaError_t BNSparseMLPKernel_dispatch(KernelParams* params) {
    const uint32_t BLOCK_M = KernelTraits::BLOCK_M;
    const uint32_t BLOCK_N = KernelTraits::BLOCK_N;
    const uint32_t BLOCK_K = KernelTraits::BLOCK_K;

    using dot_dtype = typename KernelTraits::dot_dtype;

    const uint32_t M = KernelTraits::M;
    const uint32_t N = KernelTraits::N;
    const uint32_t K = KernelTraits::K;
    const uint32_t G = KernelTraits::G;
    const uint32_t NG = KernelTraits::NG;
    const uint32_t num_stages = KernelTraits::num_stages;
    const uint32_t num_threads = KernelTraits::num_threads;

    const dim3 grid_size(CDIV(M, BLOCK_M), CDIV(N, BLOCK_N), 1);
    const dim3 block_size(num_threads, 1, 1);

    auto kernel = BNSparseMLPKernel<KernelTraits, KernelParams>;
    const size_t smem_size = sizeof(SharedMem<dot_dtype, BLOCK_M, BLOCK_N, BLOCK_K, num_stages>);
    return cudaLaunchKernel((void*)kernel, grid_size, block_size, params, smem_size);
}