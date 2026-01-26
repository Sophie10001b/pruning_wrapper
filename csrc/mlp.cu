#pragma once

#include "mlp.cuh"

at::Tensor BNSparseMLPKernel_launch(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& route_mask, const at::Tensor& route_indices
) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT(b);
    CHECK_INPUT(route_mask);
    CHECK_INPUT(route_indices);

    auto o = at::empty_like(x);
    const uint32_t B = x.size(0);
    const uint32_t L = x.size(1);
    const uint32_t M = B * L;
    const uint32_t K = x.size(2);
    const uint32_t N = w.size(0);
    const uint32_t NG = route_mask.size(0);
    const uint32_t G = CDIV(N, NG);

    assert(x.size(2) == w.size(1));

    const uint32_t BLOCK_M = 32;
    const uint32_t BLOCK_N = G;
    const uint32_t BLOCK_K = 64;
    
    DISPATCH_TORCH_TO_NV_WITH_UINT32(x.scalar_type(), nv_float_type, nv_int_type, [&] {
        auto x_ptr = x.data_ptr<nv_float_type>();
        auto w_ptr = w.data_ptr<nv_float_type>();
        auto b_ptr = b.data_ptr<nv_float_type>();
        auto o_ptr = o.data_ptr<nv_float_type>();
        auto route_mask_ptr = route_mask.data_ptr<uint8_t>();
        auto route_indices_ptr = route_indices.data_ptr<nv_int_type>();

        const uint32_t vec_size = 128 / sizeof(nv_float_type);

        using params_type = KernelParams<nv_float_type, nv_int_type>;
        using traits_type = KernelTraits<nv_float_type, float, nv_int_type, M, N, K, G, NG, BLOCK_M, BLOCK_N, BLOCK_K, vec_size, 3>;

        cudaError_t status = BNSparseMLPKernel_dispatch<traits_type, params_type>(
            params_type{x_ptr, w_ptr, b_ptr, o_ptr, route_mask_ptr, route_indices_ptr}
        );
        return true;
    });
}



int main() {
    return 0;
}