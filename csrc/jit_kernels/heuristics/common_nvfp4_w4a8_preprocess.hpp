#pragma once
#include <torch/torch.h>

namespace deep_gemm::w4a8 {

static torch::Tensor preprocess_scales(const torch::Tensor& scale) {
    TORCH_CHECK(scale.dim() == 3 && scale.is_contiguous() && scale.size(2) % 64 == 0,
                "NVFP4 scales must be contiguous [E,K/16,N], with N divisible by 64");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat8_e4m3fn,
                "NVFP4 scales require E4M3FN");
    const auto values = scale.to(torch::kFloat);
    TORCH_CHECK(torch::isfinite(values).all().item<bool>() && values.ge(0).all().item<bool>(),
                "NVFP4 block scales must be finite and nonnegative");
    const auto bytes = scale.view(torch::kUInt8);
    return bytes.view({scale.size(0), scale.size(1), -1, 8, 8})
                .transpose(-1,-2).contiguous().view(scale.sizes());
}

static torch::Tensor ppu_w4a8_preprocess_weights(const torch::Tensor& input) {
    using namespace torch::indexing;
    TORCH_CHECK(input.scalar_type() == torch::kUInt8 and input.dim() == 3 and input.is_contiguous(),
                "NVFP4 E2M1 codes must be contiguous uint8 [E,K,N]");
    const int64_t e = input.size(0), k = input.size(1), n = input.size(2);
    TORCH_CHECK(k % 32 == 0 and n % 64 == 0, "NVFP4 W4A8 requires K divisible by 32 and N by 64");
    TORCH_CHECK(input.le(15).all().item<bool>(), "E2M1 codes must be in [0,15]");
    std::vector<int64_t> n_index(256), k_index(256 * 8);
    constexpr int nibble_offsets[] = {0, 1, 2, 3, 16, 17, 18, 19};
    for (int col = 0; col < 256; ++col) {
        n_index[col] = col % 8 * 8 + col / 32;
        for (int nibble = 0; nibble < 8; ++nibble)
            k_index[col * 8 + nibble] = (col / 8 % 4) * 4 + nibble_offsets[nibble];
    }
    const auto cpu_options = torch::TensorOptions().dtype(torch::kInt64);
    const auto n_order = torch::from_blob(n_index.data(), {256, 1}, cpu_options).to(input.device());
    const auto k_order = torch::from_blob(k_index.data(), {256, 8}, cpu_options).to(input.device());
    const auto tiles = input.view({e, k / 32, 32, n / 64, 64}).permute({0, 1, 3, 2, 4});
    const auto ordered = tiles.index({Ellipsis, k_order, n_order}).to(torch::kInt64);
    auto packed = torch::zeros({e, k / 32, n / 64, 256}, input.options().dtype(torch::kInt64));
    for (int nibble = 0; nibble < 4; ++nibble) {
        const auto low = ordered.index({Ellipsis, nibble}), high = ordered.index({Ellipsis, nibble + 4});
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(low, 7), 4 * nibble));
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(high, 7), 16 + 4 * nibble));
    }
    for (int nibble = 0; nibble < 4; ++nibble) {
        const auto low = ordered.index({Ellipsis, nibble}), high = ordered.index({Ellipsis, nibble + 4});
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(low, 8), 4 + 8 * nibble));
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(high, 8), 8 * nibble));
    }
    return packed.to(torch::kInt32).reshape({e, k / 32, 4 * n}).contiguous();
}
// Checkpoint layout matches the unified preprocessing API: low nibble first
// along K, packed uint8 [E,N,K/2]. Preparation never materializes FP8 weights.
static std::pair<torch::Tensor, torch::Tensor> preprocess_fp4_weights_and_scales(
        const torch::Tensor& weight, const torch::Tensor& scales, bool fp8_layout) {
    using namespace torch::indexing;
    TORCH_CHECK(weight.dim() == 3 && weight.scalar_type() == torch::kUInt8,
                "W4A8 packed weights must be uint8 [E,N,K/2]");
    const int64_t e = weight.size(0), n = weight.size(1), k = weight.size(2) * 2;
    const int group_size = 16;
    TORCH_CHECK(e > 0 && n > 0 && n % 64 == 0 && k > 0 && k % 32 == 0,
                "W4A8 requires E>0, N divisible by 64 and K divisible by 32");
    TORCH_CHECK(scales.device() == weight.device() &&
                scales.sizes() == std::vector<int64_t>({e, n, k / group_size}) &&
                scales.scalar_type() == (torch::kFloat8_e4m3fn),
                "W4A8 requires matching [E,N,K/group_size] native scales");
    const auto values = scales.to(torch::kFloat);
    TORCH_CHECK(torch::isfinite(values).all().item<bool>() && values.ge(0).all().item<bool>(),
                "NVFP4 block scales must be finite and nonnegative");
    if (!fp8_layout) {
        const auto low = torch::bitwise_and(weight, 15);
        const auto high = torch::bitwise_right_shift(weight, 4);
        const auto codes = torch::stack({low, high}, -1).reshape({e, n, k})
                                .transpose(1, 2).contiguous();
        return {ppu_w4a8_preprocess_weights(codes),
                preprocess_scales(scales.transpose(1, 2).contiguous())};
    }

    // Each int32 holds eight consecutive K values. Magnitudes form two
    // four-nibble byte-permute selectors; signs occupy the result byte MSBs.
    const auto bytes = weight.contiguous().view({e, n, k / 8, 4}).to(torch::kInt32);
    auto packed = torch::zeros({e, n, k / 8}, weight.options().dtype(torch::kInt32));
    for (int i = 0; i < 8; ++i) {
        const auto code = torch::bitwise_and(
            torch::bitwise_right_shift(bytes.index({Ellipsis, i / 2}), (i % 2) * 4), 15);
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(code, 7), i * 4));
        packed.bitwise_or_(torch::bitwise_left_shift(torch::bitwise_and(code, 8),
                                                    (i < 4 ? 4 : 0) + (i % 4) * 8));
    }
    const auto encoded = scales.contiguous().view(torch::kUInt8);
    return {packed, encoded};
}

} // namespace deep_gemm::w4a8
