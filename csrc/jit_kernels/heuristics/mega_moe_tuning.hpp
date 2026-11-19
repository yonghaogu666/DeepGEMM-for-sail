#pragma once

#include <array>
#include <initializer_list>
#include <optional>
#include <string>
#include <deep_gemm/common/types.cuh>

namespace deep_gemm::mega {

enum class Precision { W4A16, W4A8, FP8Channelwise, FP8Blockwise };
using Tile = std::array<int, 6>; // BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, stages

struct TuningWorkload {
    Precision precision;
    int ranks, experts, topk, hidden, intermediate, tokens;
    std::string activation;
    int capacity = 0, num_sms = 0;
};

struct BlockwiseGemm1Tuning {
    PullSchedule pull = PullSchedule::Direct;
    Gemm1WorkSchedule work = Gemm1WorkSchedule::Dynamic;
    int block_n = 256, quant_lanes = 8, stages = 4, work_batch = 2;
    bool write_padded_rows = true;
};

struct BlockwiseScatterTuning {
    int scratch_rows = 0, num_sms = 0;
    bool warp_schedule = false, reuse_smem = false;
};

struct CombineTuning {
    bool push = false, preload_routes = false;
};

struct Gemm1Tuning {
    std::optional<Tile> tile{};
    std::optional<int> block_m{};
    std::optional<int> pull_ctas{};
    BlockwiseGemm1Tuning blockwise{};
};

struct Gemm2Tuning {
    std::optional<Tile> tile{};
    // Optional applicability limits for measured scatter tiles. G1 is unaffected.
    std::optional<int> capacity{};
    bool requires_no_replicas = false;
    BlockwiseScatterTuning blockwise{};
};

struct BufferTuning {
    std::optional<int> block_m{};
};

struct TuningConfig {
    TuningWorkload workload;
    std::string profile;
    Gemm1Tuning gemm1;
    Gemm2Tuning gemm2;
    BufferTuning buffer;
    CombineTuning combine{};
};

// These are measured overrides, not complete kernel configurations. Empty
// fields use the family-specific selector. Tokens match exactly unless an
// entry explicitly retains measured buckets with inclusive upper bounds.
struct TuningCase {
    int tokens;
    Gemm1Tuning gemm1{};
    Gemm2Tuning gemm2{};
    BufferTuning buffer{};
    bool preload_routes = false;
};

struct TuningEntry {
    const char* model;
    Precision precision;
    int ranks, experts, topk, hidden, intermediate;
    const char* activation;
    std::initializer_list<TuningCase> cases;
    // Only the W4A8 profiles use their existing measured token buckets.
    // Inputs above the final bound fall back to the family selector.
    bool token_buckets = false;
    int capacity = 0, num_sms = 0;  // Zero matches any workspace or CU count.
    bool push_combine = false;
};

inline TuningConfig select_tuning(const TuningWorkload& w) {
    TuningConfig result{w, {}, {}, {}, {}};
    using P = Precision;
    static const Tile small16{16, 256, 128, 16, 64, 2};
    static const Tile small32{32, 256, 128, 32, 64, 2};
    static const Tile reuse128{128, 256, 128, 128, 32, 2};
    static const Tile scatter128{128, 256, 128, 64, 64, 2};
    static const Tile channelwise_g1_decode{128, 256, 128, 32, 64, 5};
    // Transport padding remains independent of the GEMM1 compute tile.
    static const std::initializer_list<TuningCase> k3_ep32 = {
        {4, {small16, {}, 36}, {small16}, {16}},    {8, {small16, {}, 36}, {small16}, {16}},
        {16, {small16, {}, 36}, {small16}, {16}},   {32, {small32, {}, 36}, {small32}, {32}},
        {64, {reuse128, {}, 18}, {small32}, {16}},  {128, {reuse128, {}, 18}, {scatter128}, {16}},
        {1024, {reuse128}, {}, {64}}, {2048, {reuse128}, {}, {64}},
        {4096, {reuse128}, {}, {64}},
    };
    static const Tile w4a8_g1_decode{16, 256, 128, 16, 64, 2};
    static const Tile w4a8_g2_decode{16, 256, 128, 16, 64, 2};
    static const Tile w4a8_g2_decode3{16, 256, 128, 16, 64, 3};
    static const Tile w4a8_g1_small{16, 1024, 128, 16, 64, 3};
    static const Tile w4a8_g1_medium{32, 1024, 128, 32, 64, 3};
    static const Tile w4a8_g1_large{64, 1024, 128, 64, 64, 3};
    static const Tile w4a8_g2_small{16, 512, 128, 16, 64, 3};
    static const Tile w4a8_g2_medium{32, 512, 128, 32, 64, 3};
    static const Tile w4a8_g2_large{64, 512, 128, 64, 64, 3};
    static const Tile w4a8_native160{160, 256, 128, 160, 32, 3};
    // MiMo V3 measurements use 39 CUs and separate decode/prefill workspaces.
    const auto decode_g1 = [](int rows, int pulls, BlockwiseGemm1Tuning policy,
                              PullSchedule schedule = PullSchedule::Compact) {
        policy.pull = schedule;
        policy.stages = 3;
        policy.work_batch = 1;
        policy.write_padded_rows = false;
        return Gemm1Tuning{{}, rows, pulls, policy};
    };
    const auto scatter = [](Tile tile, BlockwiseScatterTuning policy) {
        return Gemm2Tuning{tile, {}, false, policy};
    };
    static constexpr Tile bm32n256s3{32, 256, 128, 16, 64, 3};
    static constexpr Tile bm32n256s4{32, 256, 128, 16, 64, 4};
    static constexpr Tile bm64n128s3{64, 128, 128, 32, 32, 3};
    static constexpr Tile bm128n256s4{128, 256, 128, 32, 64, 4};
    static constexpr BlockwiseScatterTuning warp_schedule{.warp_schedule = true};
    static constexpr BlockwiseScatterTuning warp_schedule16{
        .scratch_rows = 16, .warp_schedule = true};
    static constexpr BlockwiseGemm1Tuning paired_compact_experts{
        .work = Gemm1WorkSchedule::DynamicSingleTile, .block_n = 128, .quant_lanes = 32};
    static const TuningEntry entries[] = {
        {"mimo_v3_decode", P::FP8Blockwise, 8, 384, 8, 2048, 5120, "situ",
         {{16, decode_g1(32, 16, paired_compact_experts), scatter(bm32n256s3, warp_schedule)},
          {32, decode_g1(32, 16, paired_compact_experts), scatter(bm32n256s3, warp_schedule)},
          {64, decode_g1(32, 24, {.block_n = 128, .quant_lanes = 16}), scatter(bm32n256s3, warp_schedule)},
          {128, decode_g1(64, 12, {}), scatter({64, 256, 128, 32, 64, 4}, warp_schedule)}}, false, 128, 39},
        {"mimo_v3_decode", P::FP8Blockwise, 16, 384, 8, 2048, 5120, "situ",
         {{16, decode_g1(32, 12, {.block_n = 128, .quant_lanes = 32}),
               scatter(bm32n256s4, warp_schedule)},
          {32, decode_g1(64, 8, {.work = Gemm1WorkSchedule::Static, .quant_lanes = 32}),
               scatter(bm32n256s4, warp_schedule)},
          {64, decode_g1(64, 12, {.work = Gemm1WorkSchedule::Static}),
               scatter({32, 128, 128, 16, 32, 3}, {.warp_schedule = true, .reuse_smem = true})},
          {128, decode_g1(64, 12, {.work = Gemm1WorkSchedule::Static}), scatter(bm128n256s4, warp_schedule)}},
         false, 128, 39, true},
        {"mimo_v3_decode", P::FP8Blockwise, 32, 384, 8, 2048, 5120, "situ",
         {{16, decode_g1(64, 8, {.quant_lanes = 32}), scatter(bm64n128s3, warp_schedule16)},
          {32, decode_g1(64, 16, {}, PullSchedule::Pipelined),
               scatter(bm64n128s3, warp_schedule16), {}, true},
          {64, decode_g1(64, 12, {.quant_lanes = 16}), scatter(bm64n128s3, {.scratch_rows = 16})},
          {128, {{}, {}, {}, {.quant_lanes = 16, .write_padded_rows = false}},
                scatter(bm128n256s4, {.num_sms = 32, .warp_schedule = true})}},
         false, 128, 39, true},
        {"mimo_v3_prefill", P::FP8Blockwise, 8, 384, 8, 2048, 5120, "situ",
         {{1024, {{}, {}, 8}, {Tile{192, 256, 128, 48, 64, 4}}},
          {2048},
          {4096, {}, {Tile{192, 256, 128, 48, 64, 3}}},
          {8192, {{}, {}, 4}, {Tile{192, 256, 128, 48, 64, 3}}}},
         false, 8192, 39},
        {"kimi_k3", P::W4A8, 8, 896, 16, 3584, 3072, "situ",
         {{4, {w4a8_g1_decode}, {w4a8_g2_decode}},
          {8, {w4a8_g1_decode}, {w4a8_g2_decode3}},
          {64, {w4a8_g1_decode}, {w4a8_g2_decode}},
          {128, {Tile{32, 512, 128, 32, 64, 3}}, {Tile{32, 256, 128, 32, 64, 3}}},
          {1024},
          {2048, {w4a8_native160}, {w4a8_native160}}}, true},
        {"kimi_k3", P::W4A8, 16, 896, 16, 3584, 3072, "situ",
         {{32, {w4a8_g1_decode}, {w4a8_g2_decode}}}},
        {"kimi_k3", P::W4A8, 16, 896, 16, 3584, 3072, "situ",
         {{16, {w4a8_g1_small}, {w4a8_g2_small}},
          {64, {w4a8_g1_medium}, {w4a8_g2_medium}},
          {4096, {w4a8_g1_large}, {w4a8_g2_large}}}, true},
        {"kimi_k3", P::W4A8, 32, 896, 16, 3584, 3072, "situ",
         {{8, {Tile{16, 512, 128, 16, 64, 2}, {}, 12}, {w4a8_g2_decode}},
          {16, {Tile{16, 512, 128, 16, 64, 3}}, {w4a8_g2_decode}},
          {32, {Tile{32, 512, 128, 32, 64, 3}, {}, 12}, {Tile{32, 256, 128, 32, 64, 3}}},
          {64, {w4a8_g1_large, {}, 12}, {Tile{64, 256, 128, 64, 64, 2}}},
          {128, {Tile{64, 512, 128, 64, 64, 2}, {}, 12}, {w4a8_g2_large}},
          {4096, {w4a8_g1_large}, {w4a8_g2_large}}}, true},

        {"kimi_k3",
         P::W4A16,
         8,
         896,
         16,
         3584,
         3072,
         "situ",
         {{1024, {reuse128}, {}, {64}},
          {2048, {reuse128}, {}, {64}},
          {4096, {reuse128}, {}, {64}}}},
        {"kimi_k3",
         P::W4A16,
         16,
         896,
         16,
         3584,
         3072,
         "situ",
         {{1024, {reuse128}}, {2048, {reuse128}},
          {4096, {reuse128}, {}, {256}}}},
        {"kimi_k3", P::W4A16, 32, 896, 16, 3584, 3072, "situ", k3_ep32},
        {"kimi_k3", P::W4A16, 32, 928, 16, 3584, 3072, "situ", k3_ep32},
        {"minimax_m3",
         P::FP8Channelwise,
         8,
         128,
         4,
         6144,
         3072,
         "swiglu_oai",
         {{32, {channelwise_g1_decode}},
          {64, {channelwise_g1_decode}},
          {128, {channelwise_g1_decode}},
          {1024, {{}, {}, 3}}, {2048, {{}, {}, 3}}, {4096, {{}, {}, 3}}}},
        {"minimax_m3_blockwise",
         P::FP8Blockwise,
         8,
         384,
         4,
         7168,
         5120,
         "swiglu_oai",
         {{32, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {64, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {128, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {1024, {{}, {}, 4}}, {4096, {{}, {}, 4}}}},
        {"minimax_m3_blockwise",
         P::FP8Blockwise,
         16,
         384,
         4,
         7168,
         5120,
         "swiglu_oai",
         {{32, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {64, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {128, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}}}},
        {"minimax_m3_blockwise",
         P::FP8Blockwise,
         32,
         384,
         4,
         7168,
         5120,
         "swiglu_oai",
         {{32, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {64, {{}, 32}, {Tile{32, 256, 128, 16, 64, 3}}},
          {1024, {{}, {}, 4}}, {4096, {{}, {}, 3}}}},
        {"glm52",
         P::FP8Channelwise,
         8,
         256,
         8,
         6144,
         2048,
         "swiglu",
         {{32, {{}, 128}, {Tile{128, 256, 128, 32, 64, 4}, 4096, true}},
          {64, {channelwise_g1_decode}},
          {128, {channelwise_g1_decode}},
          {1024, {{}, {}, 9}},
          {2048, {{}, {}, 9}},
          {4096, {{}, {}, 9}}}},
        {"glm53_flash",
         P::FP8Channelwise,
         4,
         288,
         8,
         4096,
         2048,
         "swiglu",
         {{32, {Tile{16, 128, 128, 16, 32, 4}, {}, 3}, {Tile{16, 256, 64, 16, 32, 3}}},
          {64, {Tile{32, 256, 128, 16, 32, 5}, {}, 3}, {Tile{16, 256, 64, 16, 32, 3}}},
          {128, {Tile{32, 256, 128, 16, 32, 5}, {}, 3}, {Tile{32, 256, 64, 32, 64, 4}}},
          {1024, {{}, 128, 6}, {Tile{128, 256, 128, 32, 64, 3}}}, {2048}, {4096}}},
        {"qwen38_flash",
         P::FP8Channelwise,
         4,
         512,
         10,
         2560,
         640,
         "swiglu",
         {{32, {Tile{16, 128, 128, 16, 32, 4}, {}, 6}, {Tile{16, 256, 64, 16, 64, 2}}},
          {64, {Tile{32, 256, 128, 16, 32, 5}, {}, 6}, {Tile{16, 256, 64, 16, 64, 2}}},
          {128, {Tile{32, 256, 128, 16, 32, 5}, {}, 6}, {Tile{16, 256, 64, 16, 64, 2}}},
          {1024, {{}, {}, 9}, {Tile{128, 256, 128, 32, 64, 3}}}, {2048, {{}, {}, 12}}, {4096, {{}, {}, 12}}}},
    };
    for (const auto& entry : entries) {
        if (w.precision != entry.precision || w.ranks != entry.ranks || w.experts != entry.experts ||
            w.topk != entry.topk || w.hidden != entry.hidden || w.intermediate != entry.intermediate ||
            (entry.capacity && w.capacity != entry.capacity))
            continue;
        // Every rank uses the workspace's protocol, even when local kernel choices differ.
        result.combine.push |= entry.push_combine;
        if (!result.profile.empty() || w.activation != entry.activation || (entry.num_sms && w.num_sms != entry.num_sms))
            continue;
        for (const auto& c : entry.cases) {
            if (w.tokens == c.tokens || (entry.token_buckets && w.tokens <= c.tokens)) {
                result.profile = entry.model;
                result.gemm1 = c.gemm1;
                result.gemm2 = c.gemm2;
                result.buffer = c.buffer;
                result.combine.preload_routes = c.preload_routes;
                break;
            }
        }
    }
    return result;
}

} // namespace deep_gemm::mega
