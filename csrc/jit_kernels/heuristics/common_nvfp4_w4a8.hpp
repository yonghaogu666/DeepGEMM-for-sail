#pragma once
#include "common_w4a16.hpp"

namespace deep_gemm::w4a8 {
using Config = deep_gemm_w4a16_common::W4A16Config;
using ConfigTuple = std::tuple<int, int, int, int, int, int, int, int>;

static bool is_shared_decode_tile(const Config& c) {
    return c.block_k == 128 && c.warp_k == 128 && c.warp_n == 32 &&
        ((c.warp_m == 64 && ((c.block_m == 128 && c.block_n == 128) ||
                            (c.block_m == 256 && c.block_n == 64))) ||
         (c.block_m == 64 && c.block_n == 64 && c.warp_m == 32));
}

static const char* cpp_gemm_type_name(GemmType type) {
    switch (type) {
        case GemmType::Normal: return "Normal";
        case GemmType::MGroupedMasked: return "MGroupedMasked";
        case GemmType::MGroupedNoPad: return "MGroupedNoPad";
        case GemmType::MGroupedFused: return "MGroupedFused";
        default: TORCH_CHECK(false, "Unsupported W4A8 GEMM type");
    }
}

static Config config(GemmType type, int expected_m, int n, int k,
                     const std::optional<ConfigTuple>& requested = std::nullopt,
                     int a_scale_group = 0, int b_scale_block = 0) {
    TORCH_CHECK(b_scale_block == 0 || b_scale_block == 128,
                "rhs_scale_block_size must be 0 (per channel) or 128 (128x128 blocks)");
    Config c{};
    if (requested) {
        std::tie(c.num_sms, c.block_m, c.block_n, c.block_k,
                 c.warp_m, c.warp_n, c.warp_k, c.num_stages) = *requested;
    } else {
        c = deep_gemm_w4a16_common::get_best_configs(expected_m, n, k, 1, get_num_sms(),
               type, deep_gemm_w4a16_common::W4A16Type::mxfp4_e8m0);
        // Scaled partial sums use an extra FP32 accumulator.
        c.block_m = c.warp_m = 16;
        const bool wide_k = k >= 1024;
        if (wide_k) {
            if (expected_m < 12) {
                c.block_n = c.warp_n = c.warp_k = 64; c.block_k = 256;
            } else {
                c.block_n = 256; c.warp_n = 64; c.block_k = c.warp_k = 128;
            }
        } else {
            c.block_n = 256; c.warp_n = c.block_k = c.warp_k = 64;
        }
        const bool aligned_scales = b_scale_block == 128 && a_scale_group == 128 &&
            k > 512 && k % 128 == 0;
        // Decode batches keep a narrow N tile for enough independent work.
        // Two K128 warp groups reuse each outer scale across the full K128
        // partial sum, instead of scaling four independent K64 partial sums.
        if (type == GemmType::MGroupedNoPad && aligned_scales && expected_m < 12 &&
                n >= 4096 && k >= 4096 && k % 256 == 0) {
            c.warp_n = 32;
            c.warp_k = 128;
        }
        // Medium expert batches fit M32 and avoid repeating B decode in
        // multiple M16 tiles. Stage K128 scales with the operands below.
        if (type == GemmType::MGroupedNoPad && aligned_scales &&
                expected_m >= 16 && expected_m < 32 && n >= 4096 && k >= 4096) {
            c.block_m = c.warp_m = 32;
            c.block_n = 128; c.warp_n = 32;
        }
        if (b_scale_block > 0 && (expected_m >= 96 ||
                (type == GemmType::MGroupedNoPad && aligned_scales && expected_m >= 32))) {
            // The aligned K128 path shares scale work across MMA fragments,
            // making M64 viable without spilling its two FP32 accumulators.
            c.block_m = c.warp_m = aligned_scales ? 64 : 32;
            c.block_n = 128; c.warp_n = 32;
        }
        if (n < c.block_n) c.block_n = n >= 128 ? 128 : 64;
        c.num_stages = k <= 512 || wide_k ? 2 : 3;
        // Staging scales keeps three operand stages within the M32/M64 resource
        // budget and avoids repeating global scale loads in the compute warps.
        if (aligned_scales && (c.block_m == 32 || c.block_m == 64) && c.block_n % 128 == 0 &&
                type != GemmType::MGroupedFused) c.num_stages = 3;
        // Shared decoding amortizes B conversion across two M64 warp groups.
        // Keep smaller expert batches on M64 to limit partially filled tiles.
        if (type == GemmType::MGroupedNoPad && aligned_scales &&
                expected_m >= 256 && n >= 4096 && k >= 4096) {
            c.block_m = 128; c.warp_m = 64;
            c.block_n = 128; c.warp_n = 32; c.num_stages = 3;
        }
        // A narrow N tile keeps two CTAs resident while four M64 warp
        // groups share decoded B. Smaller expert batches waste too many
        // padded rows on M256, so retain M128 below this range.
        if (type == GemmType::MGroupedNoPad && aligned_scales &&
                expected_m >= 640 && n >= 4096 && k >= 4096) {
            c.block_m = 256; c.warp_m = 64;
            c.block_n = 64; c.warp_n = 32; c.num_stages = 3;
        }
    }
    const bool shared_decode = a_scale_group == 128 && b_scale_block == 128 &&
        type != GemmType::MGroupedFused && k % 128 == 0 &&
        is_shared_decode_tile(c);
    TORCH_CHECK(c.num_sms > 0 && c.num_sms <= get_num_sms() &&
                (c.block_m == 16 || c.block_m == 32 || c.block_m == 64 || c.block_m == 128 ||
                 (c.block_m == 256 && shared_decode)) &&
                (c.warp_m == c.block_m || shared_decode) && (c.warp_n == 32 || c.warp_n == 64) &&
                c.block_n >= 64 && c.block_n % 64 == 0 && c.block_k >= 64 && c.block_k % 32 == 0 &&
                c.warp_k >= 32 && c.warp_k % 32 == 0 && c.block_n % c.warp_n == 0 &&
                c.block_k % c.warp_k == 0 && c.num_stages >= 2 && c.num_stages <= 4,
                "Invalid W4A8 tile configuration");
    const int warps_on_k = c.block_k / c.warp_k;
    TORCH_CHECK((warps_on_k & (warps_on_k - 1)) == 0,
                "W4A8 K-warp reduction requires a power of two number of warp groups");
    c.n_expand = !shared_decode && k <= 512 && n % (c.block_n * 4) == 0 && k % c.block_k == 0 &&
                 c.num_stages == 2 && c.block_k == c.warp_k ? 4 : 1;
    return c;
}

static int smem_size(const Config& c, int group_size, int b_scale_block = 0,
                     int a_scale_group = 0, int k = 0, bool fused = false) {
    const int a = c.block_m * c.block_k * c.num_stages;
    const int b = c.block_k * c.block_n / 2 * c.num_stages;
    const int s = c.block_k / group_size * c.block_n * c.num_stages;
    const int reduce = 16 * 64 * (c.block_n / c.warp_n) * (c.block_k / c.warp_k) * 4;
    const int epilogue = 16 * c.block_n * 2;
    const int lut = group_size == 16 && b_scale_block > 0 ? 1024 : 256;
    const bool staged_scales = group_size == 16 && a_scale_group == 128 && b_scale_block == 128 &&
        c.block_k == 128 && c.warp_k == 128 && k % 128 == 0 && c.n_expand == 1 &&
        !fused && c.block_m >= 32 &&
        (c.block_n % 128 == 0 || (c.block_m == 256 && c.block_n == 64) ||
         (c.block_m == 64 && c.block_n == 64 && c.warp_m == 32));
    const int outer_scales = staged_scales ? (c.block_m + 32) * c.num_stages * sizeof(float) : 0;
    const int decoded = staged_scales && is_shared_decode_tile(c) ? c.block_n * c.block_k : 0;
    return lut + outer_scales + decoded + std::max({a + b + s, reduce, epilogue});
}

} // namespace deep_gemm::w4a8
