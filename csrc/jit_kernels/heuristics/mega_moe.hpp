#pragma once
#include <deep_gemm/layout/mega_moe_types.cuh>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>
#include "common_fp4.hpp"
#include "mega_moe_tuning.hpp"

namespace deep_gemm::mega_moe_config {
struct DispatchConfig {
    // Dispatch and GEMM1 row granularity.
    int block_m = 0;
    float activation_clamp = 0;
    bool fast_math = true;
    bool gemm1_dynamic_queue = false;
    bool fused_situ = false;
    bool fused_swiglu = false;
    int gemm2_block_m = 0;
    int gemm1_block_k = 0;
    int gemm1_block_n = 0;
    int gemm1_n = 0;
    int gemm1_stages = 0;
    int gemm1_warp_m = 0;
    int gemm1_warp_n = 0;
    int gemm_sms = 0;
    int nmtpr = 0;
    int num_experts = 0;
    int num_experts_per_rank = 0;
    int num_ranks = 0;
    int num_sms = 0;
    int num_threads = 0;
    int num_topk = 0;
    int pull_slots = 0;
    int pull_sms = 0;
    int sfb_layout_block_n = 0;
    int sf_bytes = 0;
    bool silu_then_clamp = false;
    float situ_beta = 0;
    float situ_linear_beta = 0;
    int token_bytes = 0;
};

struct PreDispatchConfig {
    int block_threads = 0;
    int hidden = 0;
    int num_sms = 0;
    int num_topk = 0;
};

struct ScatterConfig {
    int block_k = 0;
    int block_m = 0;
    int block_n = 0;
    int k = 0;
    bool fp8_output = false;
    int fp8_exp_shift = 0;
    mega_moe::Fp8ScaleMode fp8_scale_mode = mega_moe::Fp8ScaleMode::Fixed;
    int n = 0;
    int num_groups = 0;
    int num_master_groups = 0;
    int m_tiles_per_group = 2;
    int num_stages = 0;
    int warp_m = 0;
    int warp_n = 0;
    int world_size = 0;
};

inline constexpr std::pair<const char*, int DispatchConfig::*> dispatch_config_fields[] = {
    {"NUM_RANKS", &DispatchConfig::num_ranks},
    {"NUM_SMS", &DispatchConfig::num_sms},
    {"NUM_THREADS", &DispatchConfig::num_threads},
    {"NUM_EXPERTS", &DispatchConfig::num_experts},
    {"NUM_EXPERTS_PER_RANK", &DispatchConfig::num_experts_per_rank},
    {"NUM_TOPK", &DispatchConfig::num_topk},
    {"BLOCK_M", &DispatchConfig::block_m},
    {"NMTPR", &DispatchConfig::nmtpr},
    {"TOKEN_BYTES", &DispatchConfig::token_bytes},
    {"SF_BYTES", &DispatchConfig::sf_bytes},
    {"GEMM1_N", &DispatchConfig::gemm1_n},
    {"GEMM1_BLOCK_N", &DispatchConfig::gemm1_block_n},
    {"GEMM1_BLOCK_K", &DispatchConfig::gemm1_block_k},
    {"GEMM1_WARP_M", &DispatchConfig::gemm1_warp_m},
    {"GEMM1_WARP_N", &DispatchConfig::gemm1_warp_n},
    {"GEMM1_STAGES", &DispatchConfig::gemm1_stages},
};

using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;
using Overrides = std::map<std::string, int>;

inline int integer(const char* name, int fallback) {
    return get_env<int>(name, fallback);
}

inline TuningConfig tuning_config(const TuningWorkload& workload) {
    return select_tuning(workload, device_runtime->get_prop()->name, get_num_sms());
}

inline std::tuple<int, int, int> fp4_smem(int stages, int bm, int bn, int wm, int wn, int bk = 128,
                                          int bpp = 1) {
    TORCH_CHECK(stages > 0 && bm > 0 && bn > 0 && bk > 0 && bpp > 0);
    const auto [sfa, invalid_a] = deep_gemm_fp4_common::get_sf_per_stage_size(bm, bk);
    const auto [sfb, invalid_b] = deep_gemm_fp4_common::get_sf_per_stage_size(bn, bk);
    return {std::max(bm * bn * 2,
                     stages * (bm + bn) * bk * bpp + stages * (sfa + sfb) * bpp - invalid_a - invalid_b),
            128, 0};
}
inline Config fp4_config(int total_m, int m, int n, int k, int groups, int sms,
                         GemmType type = GemmType::MGroupedNoPad, int max_bn = 256) {
    auto config =
        deep_gemm_fp4_common::get_best_configs(total_m, m, n, k, groups, sms, false, false, type, max_bn);
    const auto [ns, bm, bn, bk, wm, wn, stages, smem] = config;
    std::get<7>(config) = fp4_smem(stages, bm, bn, wm, wn, bk);
    return config;
}
inline int threads(int bm, int bn, int wm, int wn) {
    TORCH_CHECK(wm > 0 && wn > 0 && bm % wm == 0 && bn % wn == 0, "Invalid MegaMoE warp tile");
    return bm / wm * (bn / wn) * 32;
}
inline int pre_dispatch_threads(int hidden) {
    for (int t = std::min(hidden / 8, 512) / 32 * 32; t >= 32; t -= 32)
        if (hidden / 8 % t == 0)
            return t;
    return 32;
}
inline int dispatch_pull_slots(int num_launch_threads, int token_bytes, int scale_bytes) {
    // Match the device pull-staging limit; pull and GEMM reuse this storage.
    constexpr int max_staging_bytes = 256 * 1024;
    TORCH_CHECK(num_launch_threads > 0 && num_launch_threads % 32 == 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0 && token_bytes <= max_staging_bytes - scale_bytes,
                "MegaMoE pull staging must fit at least one token and its scales in 256 KiB");
    return std::min({16, num_launch_threads / 32, max_staging_bytes / (token_bytes + scale_bytes)});
}
inline void apply_gemm1_tuning(DispatchConfig& c, const Gemm1Tuning& tuning,
                               const std::optional<Overrides>& overrides, int max_bn = 0, int max_bk = 0) {
    const auto set = [&](const char* key, int DispatchConfig::* field, int value) {
        if (!overrides || !overrides->count(key))
            c.*field = value;
    };
    // A workspace may impose a smaller tile/packed-scale layout than the
    // measured profile. Use the workspace's tile selector in that case.
    const bool use_tile = tuning.tile && (!max_bn || (*tuning.tile)[1] <= max_bn) &&
                          (!max_bk || (*tuning.tile)[2] <= max_bk);
    if (use_tile) {
        const auto& t = *tuning.tile;
        set("BLOCK_M", &DispatchConfig::block_m, t[0]);
        set("GEMM1_BLOCK_N", &DispatchConfig::gemm1_block_n, t[1]);
        set("GEMM1_BLOCK_K", &DispatchConfig::gemm1_block_k, t[2]);
        set("GEMM1_WARP_M", &DispatchConfig::gemm1_warp_m, t[3]);
        set("GEMM1_WARP_N", &DispatchConfig::gemm1_warp_n, t[4]);
        set("GEMM1_STAGES", &DispatchConfig::gemm1_stages, t[5]);
    }
    if (tuning.block_m)
        set("BLOCK_M", &DispatchConfig::block_m, *tuning.block_m);
    if ((use_tile || tuning.block_m) && (!overrides || !overrides->count("NUM_THREADS")))
        c.num_threads = threads(c.block_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
}
// Packed FP4 transport dimensions.
inline DispatchConfig dispatch_config(int ranks, int experts, int topk, int capacity,
                                      int token_bytes, int scale_bytes) {
    TORCH_CHECK(ranks > 0 && experts > 0 && experts % ranks == 0 && topk > 0 && capacity >= 0);
    TORCH_CHECK(token_bytes > 0 && scale_bytes >= 0);
    DispatchConfig c;
    c.num_ranks = ranks;
    c.num_sms = get_num_sms();
    c.num_experts = experts;
    c.num_experts_per_rank = experts / ranks;
    c.num_topk = topk;
    c.nmtpr = capacity;
    c.token_bytes = token_bytes;
    c.sf_bytes = scale_bytes;
    return c;
}

// Select W4A4 dispatch threads and GEMM1 tiles from the transport byte count.
inline DispatchConfig w4_dispatch_config(DispatchConfig c, int intermediate, int tokens,
                                         int bn, int bk, int wm, int wn, int stages,
                                         const std::optional<Overrides>& overrides) {
    TORCH_CHECK(intermediate > 0 && bn > 0 && bk > 0 && tokens >= 0 && tokens <= c.nmtpr);
    const int epr = c.num_experts_per_rank, bytes = c.token_bytes, sms = c.num_sms;
    bn = std::min(bn, intermediate);
    bk = std::min(bk, bytes);
    TORCH_CHECK(intermediate % bn == 0 && bytes % bk == 0 && bk % 16 == 0);
    const int max_bn = bn, total_m = std::max(1, tokens * std::min(c.num_topk, epr));
    const int em = std::max(1, ceil_div(total_m, epr));
    int bm, selected_sms;
    std::tuple<int, int, int> smem;
    std::tie(selected_sms, bm, bn, bk, wm, wn, stages, smem) =
        fp4_config(total_m, em, intermediate, bytes, epr, sms, GemmType::MGroupedNoPad, bn);
    // Apply measured persistent-GEMM corrections in shape-based bins.
    const bool wide = max_bn >= 256 && intermediate % 256 == 0 && bytes % 128 == 0;
    if (wide && epr >= sms && em >= 2 && em <= 5) {
        bm = 256;
        bn = 256;
        bk = 128;
        wm = 64;
        wn = 64;
        stages = 3;
    } else if (wide && epr <= sms / 2 && em >= 16 && em <= 32) {
        bm = 64;
        bn = 256;
        bk = 128;
        wm = 32;
        wn = 64;
        stages = 3;
    }
    if (bn < 128) {
        TORCH_CHECK(intermediate % 128 == 0 && (bm == 16 || bm == 32 || bm == 64));
        bn = 128;
        bk = 128;
        wm = bm == 16 ? 16 : 32;
        wn = 64;
        stages = bm == 16 ? 3 : 2;
    }
    TORCH_CHECK(intermediate % bn == 0 && bytes % bk == 0);
    c.num_threads = threads(bm, bn, wm, wn);
    c.block_m = bm;
    c.gemm1_n = intermediate;
    c.gemm1_block_n = bn;
    c.gemm1_block_k = bk;
    c.gemm1_warp_m = wm;
    c.gemm1_warp_n = wn;
    c.gemm1_stages = stages;
    if (overrides) {
        for (const auto& [key, val] : *overrides) {
            const auto found =
                std::find_if(std::begin(dispatch_config_fields), std::end(dispatch_config_fields),
                             [&](const auto& field) { return key == field.first; });
            TORCH_CHECK(found != std::end(dispatch_config_fields), "Unknown MegaMoE config key: ", key);
            c.*(found->second) = val;
        }
        if (!overrides->count("NUM_THREADS") &&
            (overrides->count("BLOCK_M") || overrides->count("GEMM1_BLOCK_N") ||
             overrides->count("GEMM1_WARP_M") || overrides->count("GEMM1_WARP_N")))
            c.num_threads = threads(c.block_m, c.gemm1_block_n, c.gemm1_warp_m, c.gemm1_warp_n);
    }
    return c;
}
inline DispatchConfig fp4_dispatch_config(int ranks, int experts, int topk, int hidden, int intermediate,
                                          int capacity, int tokens, int bn = 256, int bk = 128, int wm = 32,
                                          int wn = 64, int stages = 3,
                                          const std::optional<Overrides>& overrides = std::nullopt) {
    TORCH_CHECK(hidden > 0 && hidden % 32 == 0);
    return w4_dispatch_config(dispatch_config(ranks, experts, topk, capacity, hidden / 2, hidden / 32),
                              intermediate, tokens, bn, bk, wm, wn, stages, overrides);
}
// Resolve GEMM2 once, before GEMM1 produces its padded row metadata.
// Explicit workspace tiles override measured tuning in SymmBufferState.
inline Config fp4_scatter_config(int n, int groups, const TuningConfig& tuning) {
    TORCH_CHECK(groups > 0 && n > 0 && n % 256 == 0,
                "FP4 scatter requires N divisible by 256");
    Tile tile{};
    if (tuning.gemm2.tile) {
        tile = *tuning.gemm2.tile;
    } else {
        // Use actual tokens, rather than pool capacity. Pick the smallest M
        // bin covering the estimate, capped at the largest supported tile.
        const int em = std::max(1, ceil_div(tuning.workload.tokens * tuning.workload.topk, groups));
        static constexpr Tile defaults[] = {
            {16, 256, 64, 16, 64, 2},
            {32, 256, 64, 32, 32, 2},
            {64, 256, 64, 32, 64, 2},
            {128, 256, 64, 64, 64, 3},
            {256, 256, 128, 64, 64, 3},
        };
        for (const auto& candidate : defaults) {
            tile = candidate;
            if (em <= candidate[0])
                break;
        }
    }
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    TORCH_CHECK(bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256,
                "W4A4 GEMM2 metadata requires BLOCK_M in {16,32,64,128,256}");
    TORCH_CHECK(bn > 0 && bn % 256 == 0 && n % bn == 0,
                "FP4 scatter requires complete 256-column tiles");
    return {get_num_sms(), bm, bn, bk, wm, wn, stages, fp4_smem(stages, bm, bn, wm, wn, bk)};
}

// Keep dispatch scheduling independent of buffer ownership and kernel argument assembly.
inline void fp4_dispatch_schedule(DispatchConfig& c, const Gemm1Tuning& tuning = {}) {
    const int sms = c.num_sms;
    const int pulls = integer("MEGA_MOE_PULL_SMS", tuning.pull_sms.value_or(std::min(6, sms - 1)));
    TORCH_CHECK(pulls >= 1 && pulls < sms,
                "W4A4 pull CTAs must leave at least one compute CTA");
    c.gemm1_dynamic_queue = tuning.dynamic_queue.value_or(true);
    c.pull_sms = pulls;
    c.gemm_sms = sms - pulls;
    c.pull_slots = dispatch_pull_slots(c.num_threads, c.token_bytes, c.sf_bytes);
}

inline int combine_num_threads(int tokens, int hidden) {
    // One CTA per token. Use the actual local row count, not buffer capacity
    // or dispatch padding, to balance row parallelism and work per thread.
    if (tokens >= 512)
        return 256;
    if (tokens <= 64 && hidden > 4096)
        return 1024;
    return 512;
}

} // namespace deep_gemm::mega_moe_config
