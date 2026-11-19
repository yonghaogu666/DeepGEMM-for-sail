#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>

#include "mega_moe_dispatch.hpp"
#include "mega_moe_combine.hpp"
#include "w4a8_streamed_gemm.hpp"

namespace deep_gemm::mega {

using W4A8Weights = std::tuple<torch::Tensor, torch::Tensor, float>;

inline bool fuse_w4a8_group128_output(const Gemm1Config& config) {
    return config.group_activation_scales && (config.predecode_weights || config.fused_group_quantization) &&
           config.tile.block_n % 256 == 0;
}

struct FP8FP4Linear1Params {
    layout::SymBuffer<> sym_buffer;
    int num_tokens;
    const uint8_t *weights, *scales;
    float inv_scale;
    const uint8_t *replicas, *replica_scales;
    float replica_inv_scale;
    int master_experts;
    at::BFloat16* output;
    float activation_clamp, activation_alpha, situ_beta, situ_linear_beta;
    uint4* block_info;
    uint8_t* quantized_output;
    float* output_scales;
};

struct FP8FP4Linear2Params {
    const uint8_t* acts;
    const float* act_scales;
    const uint8_t *weights, *scales;
    float inv_scale;
    const uint8_t *replicas, *replica_scales;
    float replica_inv_scale;
    int master_experts;
    int32_t* rows;
    const int32_t* valid_rows;
    const int64_t* routes;
    at::BFloat16* output;
    const int64_t* peer_offsets;
    int capacity, pool_rows, rank_idx;
};

template <MegaMoEPhaseKind Stage, GatedActivation Activation = GatedActivation::SwiGLU>
class FP8FP4MegaMoEGemmRuntime final : public LaunchRuntime<FP8FP4MegaMoEGemmRuntime<Stage, Activation>> {
public:
    static_assert(Stage == MegaMoEPhaseKind::Linear1 || Stage == MegaMoEPhaseKind::Linear2,
                  "unsupported MegaMoE stage");
    static constexpr bool kIsLinear1 = Stage == MegaMoEPhaseKind::Linear1;
    using Params = std::conditional_t<kIsLinear1, FP8FP4Linear1Params, FP8FP4Linear2Params>;
    using LaunchInfo = std::conditional_t<kIsLinear1, Gemm1Config, ScatterConfig>;

    struct Args {
        const char* kernel_name;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        Params kernel_params{};
        uint64_t routing_bytes = 0;
    };

    static std::string generate_impl(const Args& args) {
        const auto& c = args.launch_info;
        if constexpr (kIsLinear1) {
            const auto& dispatch = c.dispatch;
            const auto& tile = c.tile;
            // A register-limited specialization may be rebuilt with a smaller cooperative grid.
            // Fit complete pull teams and reserve a CTA to start G1 while their input arrives.
            const int pull_group_size = std::min(dispatch.pull_group_size, c.num_ctas - 1);
            TORCH_CHECK(pull_group_size > 0, "W4A8 dispatch requires a pull CTA and a compute CTA");
            const int pull_ctas = std::min(dispatch.pull_ctas, (c.num_ctas - 1) / pull_group_size * pull_group_size);
            TORCH_CHECK(dispatch.num_ranks > 0 && dispatch.num_experts % dispatch.num_ranks == 0,
                        "invalid MegaMoE layout");
            const layout::RoutedMoEInputLayout inputs(dispatch.num_ranks, dispatch.num_experts, dispatch.nmtpr,
                                                     dispatch.num_topk, dispatch.token_bytes, dispatch.sf_bytes, args.routing_bytes);
            return fmt::format(R"(
#define DG_MOE_FAST_MATH {}
#include <deep_gemm/impls/sm89_fp8_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8Fp4Gemm1Kernel<{}, {}, {}, {}, {}, {}, deep_gemm::GatedActivation({}), deep_gemm::w4fa8::FragmentPolicy::{}, deep_gemm::w4fa8::ALoadPolicy::{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_fp4_mega_moe_l1<Builder>);
}}
)",
                int(c.fast_math), dispatch.block_m, tile.block_n, dispatch.token_bytes, tile.warp_m, tile.warp_n,
                tile.num_stages, int(Activation),
                c.fragment == w4fa8::FragmentPolicy::FullK ? "FullK" : "RollingK32",
                !w4fa8::uses_ring_a(tile.warp_m) ? "Bulk" : "Ring2", dispatch.num_experts, c.num_threads,
                dispatch.num_ranks, c.num_ctas, dispatch.num_experts_per_rank, dispatch.num_topk, c.n, pull_ctas,
                pull_group_size, c.metadata.block_m, c.metadata.valid_rows, inputs.topk_idx, inputs.input_tokens,
                inputs.input_scales, inputs.dispatched_tokens, inputs.dispatched_scales, inputs.gemm2_metadata,
                dispatch.nmtpr, args.routing_bytes, c.group_activation_scales, c.cache_group_scales, c.predecode_weights,
                fuse_w4a8_group128_output(c), c.input_schedule != W4A8InputSchedule::Wait,
                c.input_schedule == W4A8InputSchedule::OverlapWeightsUnrollK, args.launch_args.smem_size);
        } else {
            return fmt::format(R"(
#include <deep_gemm/impls/sm89_fp8_fp4_mega_moe.cuh>
using Builder = deep_gemm::MegaMoEFp8Fp4Gemm2Kernel<{}, {}, {}, {}, {}, {}, {}, {}, deep_gemm::w4fa8::FragmentPolicy::{}, deep_gemm::w4fa8::ALoadPolicy::{}, {}, {}, {}, {}, {}>;
using Kernel = typename Builder::Kernel;
static_assert(Builder::shared_memory_bytes() == {}, "MegaMoE shared storage size differs from host");
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_gemm::sm89_fp8_fp4_mega_moe_l2<Builder>);
}}
)",
                c.n, c.k, c.block_m, c.block_n, c.warp_m, c.warp_n, c.num_groups, c.num_stages,
                c.fragment == w4fa8::FragmentPolicy::FullK ? "FullK" : "RollingK32",
                !w4fa8::uses_ring_a(c.warp_m) ? "Bulk" : "Ring2", c.num_threads, c.num_ranks,
                c.group_activation_scales, c.cache_group_scales, c.predecoded_weights, args.launch_args.smem_size);
        }
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, const Args& args) {
        const auto& p = args.kernel_params;
        if constexpr (kIsLinear1) {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.sym_buffer, p.num_tokens, p.weights, p.scales, p.inv_scale, p.replicas, p.replica_scales,
                p.replica_inv_scale, p.master_experts, p.output, p.activation_clamp, p.activation_alpha,
                p.situ_beta, p.situ_linear_beta, p.block_info, p.quantized_output, p.output_scales));
        } else {
            DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config,
                p.acts, p.act_scales, p.weights, p.scales, p.inv_scale, p.replicas, p.replica_scales,
                p.replica_inv_scale, p.master_experts, p.rows, p.valid_rows, p.routes, p.output,
                p.peer_offsets, p.capacity, p.pool_rows, p.rank_idx));
        }
    }
};

template <GatedActivation Activation>
static auto make_fp8_fp4_linear1_args(
    const W4A8Weights& weights, const W4A8Weights& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int num_tokens, float activation_clamp, float activation_alpha, float situ_beta, float situ_linear_beta,
    const Gemm1Config& config) {
    using Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1, Activation>;
    const auto& [w, s, inv] = weights;
    const auto& [rw, rs, rinv] = replicas;
    return typename Runtime::Args{
        .kernel_name = "w4a8_mega_moe_dispatch_gemm1", .launch_info = config,
        .launch_args = {dim3(config.num_ctas), dim3(config.num_threads), config.tile.smem_size},
        .kernel_params = {
            .sym_buffer = sym_buffer, .num_tokens = num_tokens,
            .weights = device_pointer<uint8_t>(w), .scales = device_pointer<uint8_t>(s), .inv_scale = inv,
            .replicas = device_pointer<uint8_t>(rw), .replica_scales = device_pointer<uint8_t>(rs), .replica_inv_scale = rinv,
            .master_experts = static_cast<int>(w.size(0)),
            .output = math::advance_ptr<at::BFloat16>(base, buffer.bf16_scratch),
            .activation_clamp = activation_clamp, .activation_alpha = activation_alpha,
            .situ_beta = situ_beta, .situ_linear_beta = situ_linear_beta,
            .block_info = math::advance_ptr<uint4>(base, buffer.block_info),
            .quantized_output = base + buffer.l2_acts,
            .output_scales = math::advance_ptr<float>(base, buffer.l2_scales)},
        .routing_bytes = buffer.dispatch.routing_bytes};
}

static auto make_fp8_fp4_linear2_args(
    const W4A8Weights& weights, const W4A8Weights& replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, int rank_idx,
    const DispatchConfig& dispatch, const ScatterConfig& config) {
    using Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>;
    const auto& [w, s, inv] = weights;
    const auto& [rw, rs, rinv] = replicas;
    const layout::Gemm2Metadata meta(base + buffer.gemm2_metadata, config.num_groups);
    return Runtime::Args{
        .kernel_name = "w4a8_mega_moe_scatter", .launch_info = config,
        .launch_args = {dim3(config.num_sms), dim3(config.num_threads), config.smem_size},
        .kernel_params = {
            .acts = base + buffer.l2_acts, .act_scales = math::advance_ptr<float>(base, buffer.l2_scales),
            .weights = device_pointer<uint8_t>(w), .scales = device_pointer<uint8_t>(s), .inv_scale = inv,
            .replicas = device_pointer<uint8_t>(rw), .replica_scales = device_pointer<uint8_t>(rs), .replica_inv_scale = rinv,
            .master_experts = static_cast<int>(w.size(0)),
            .rows = scatter_layout(config.num_groups, meta.rows, math::advance_ptr<int32_t>(base, buffer.block_info)),
            .valid_rows = meta.valid_rows, .routes = math::advance_ptr<int64_t>(base, buffer.packed_routes),
            .output = math::advance_ptr<at::BFloat16>(base, buffer.combine),
            .peer_offsets = math::advance_ptr<int64_t>(base, buffer.peer_offsets),
            .capacity = dispatch.nmtpr, .pool_rows = static_cast<int>(buffer.dispatch.num_max_pool_tokens),
            .rank_idx = rank_idx}};
}

// The symmetric allocation owns the decoded buffers and stream. Its layer aliases
// execute serially and reuse the same addresses during graph replay.
struct PredecodedWorkspace {
    torch::Tensor first, second;
    int device;
    hggcStream_t stream = nullptr;
    hggcEvent_t fork = nullptr, ready = nullptr;

    PredecodedWorkspace(const torch::Tensor& owner, int64_t w1, int64_t w2)
        : first(torch::empty({w1}, owner.options())), second(torch::empty({w2}, owner.options())),
          device(owner.get_device()) {}
    PredecodedWorkspace(const PredecodedWorkspace&) = delete;
    PredecodedWorkspace& operator=(const PredecodedWorkspace&) = delete;

    void initialize() {
        int least_priority = 0, greatest_priority = 0;
        DG_HGGC_RUNTIME_CHECK(hggcDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        DG_HGGC_RUNTIME_CHECK(hggcStreamCreateWithPriority(&stream, hggcStreamNonBlocking, least_priority));
        DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&fork, hggcEventDisableTiming));
        DG_HGGC_RUNTIME_CHECK(hggcEventCreateWithFlags(&ready, hggcEventDisableTiming));
    }

    ~PredecodedWorkspace() noexcept {
        int previous = device;
        if (hggcGetDevice(&previous) != hggcSuccess || hggcSetDevice(device) != hggcSuccess)
            return;
        if (fork)
            hggcEventDestroy(fork);
        if (ready)
            hggcEventDestroy(ready);
        if (stream)
            hggcStreamDestroy(stream);
        hggcSetDevice(previous);
    }
};

using PredecodedWorkspaces = std::map<std::pair<int64_t, int64_t>, std::unique_ptr<PredecodedWorkspace>>;

inline PredecodedWorkspace& predecoded_workspace(PredecodedWorkspaces& workspaces, const torch::Tensor& owner,
                                                 const MegaMoEPrecisionConfigs& c) {
    const int64_t groups = c.l2.num_groups;
    const int64_t w1 = groups * c.l1.n * c.l1.k, w2 = groups * c.l2.n * c.l2.k;
    const auto sizes = std::make_pair(w1, w2);
    if (const auto it = workspaces.find(sizes); it != workspaces.end())
        return *it->second;

    auto next = std::make_unique<PredecodedWorkspace>(owner, w1, w2);
    next->initialize();
    return *workspaces.emplace(sizes, std::move(next)).first->second;
}

inline std::pair<W4A8Weights, W4A8Weights> predecode_moe_weights(
    const W4A8Weights& weights, const W4A8Weights& replicas, const torch::Tensor& storage,
    int n, int k, int groups, hggcStream_t stream = current_stream(), int blocks = 0) {
    const int masters = std::get<0>(weights).size(0);
    using Runtime = w4fa8::streamed::StreamedDecodeRuntime;
    auto decode = [&](const W4A8Weights& bank, int count, int offset) -> W4A8Weights {
        if (!count) return bank;
        Runtime::Args args{
            .n = n, .k = k, .tile_n = n, .groups = count,
            .launch_args = {dim3(blocks ? blocks : get_num_sms() * 8), dim3(256), 0},
            .params = {}, .skip_fence = true};
        const auto runtime = compiler->build(Runtime::kernel_name(args), Runtime::generate(args));
        // A compile-only pass needs the specialization, without storage or GPU work.
        if (!storage.defined()) return bank;
        auto output = storage.narrow(0, int64_t(offset) * n * k, int64_t(count) * n * k).view({count, n, k});
        args.params = {device_pointer<uint8_t>(std::get<0>(bank)), device_pointer<uint8_t>(std::get<1>(bank)),
                       device_pointer<uint8_t>(output), 0, nullptr, 0};
        Runtime::launch(runtime, args, stream);
        return {output, std::get<1>(bank), std::get<2>(bank)};
    };
    return {decode(weights, masters, 0), decode(replicas, groups - masters, masters)};
}

inline void fp8_fp4_mega_moe_impl(
    const torch::Tensor& y, const W4A8Weights& l1_weights, const W4A8Weights& l2_weights,
    const W4A8Weights& l1_weights_replicas, const W4A8Weights& l2_weights_replicas,
    uint8_t* base, const layout::RoutedMoEBuffer& buffer, const layout::SymBuffer<>& sym_buffer,
    int rank_idx, int num_tokens, GatedActivation activation, float activation_clamp, bool fast_math,
    float activation_alpha, float situ_beta, float situ_linear_beta,
    const MegaMoEPrecisionConfigs& configs, PredecodedWorkspace* workspace,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    using Linear2Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear2>;
    auto config = configs.l1;
    config.fast_math = fast_math;
    const int rows = buffer.dispatch.num_max_pool_tokens;
    auto first = config.predecode_weights
        ? predecode_moe_weights(l1_weights, l1_weights_replicas, workspace ? workspace->first : torch::Tensor{},
                               config.n, config.k, configs.l2.num_groups)
        : std::make_pair(l1_weights, l1_weights_replicas);
    if (workspace) {
        // Publish caller-side weight updates and the completed W1 decode.
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(workspace->fork, current_stream()));
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(workspace->stream, workspace->fork, 0));
    }
    auto launch_linear1 = [&](auto activation_tag) {
        constexpr auto kind = decltype(activation_tag)::value;
        using Runtime = FP8FP4MegaMoEGemmRuntime<MegaMoEPhaseKind::Linear1, kind>;
        const auto args = make_fp8_fp4_linear1_args<kind>(
            first.first, first.second, base, buffer, sym_buffer, num_tokens,
            activation_clamp, activation_alpha, situ_beta, situ_linear_beta, config);
        launch_mega_moe_kernel<Runtime, true>(
            args, gemm_profile("w4fa8_mma", rows, configs.expected_m, config));
    };
    if (activation == GatedActivation::SiTU)
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SiTU>{});
    else if (activation == GatedActivation::SwiGLUStep)
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SwiGLUStep>{});
    else if (activation == GatedActivation::SwiGLUOAI)
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SwiGLUOAI>{});
    else
        launch_linear1(std::integral_constant<GatedActivation, GatedActivation::SwiGLU>{});
    // GEMM1 reads W1 while the auxiliary stream fills separate W2 storage.
    // One decoder CTA per CU bounds interference with the resident GEMM1.
    auto second = config.predecode_weights
        ? predecode_moe_weights(l2_weights, l2_weights_replicas, workspace ? workspace->second : torch::Tensor{},
                               configs.l2.n, configs.l2.k, configs.l2.num_groups, workspace ? workspace->stream : nullptr, get_num_sms())
        : std::make_pair(l2_weights, l2_weights_replicas);
    if (workspace)
        DG_HGGC_RUNTIME_CHECK(hggcEventRecord(workspace->ready, workspace->stream));
    if (!fuse_w4a8_group128_output(config))
        fp8_channelwise_mega_moe_quantize_local_pool_kernel(config, sym_buffer, base, buffer);
    update_recv_stats(base, buffer, config.dispatch.num_experts_per_rank, cumulative_local_expert_recv_stats);
    // GEMM2 consumes both GEMM1's output and the decoded W2 bank.
    if (workspace)
        DG_HGGC_RUNTIME_CHECK(hggcStreamWaitEvent(current_stream(), workspace->ready, 0));
    const auto l2_args = make_fp8_fp4_linear2_args(
        second.first, second.second, base, buffer, rank_idx, config.dispatch, configs.l2);
    launch_mega_moe_kernel<Linear2Runtime, false, 8>(
        l2_args, gemm_profile("w4fa8_mma", rows, configs.expected_m, configs.l2));
    combine_mega_moe(base, buffer, config.dispatch, sym_buffer, y, rank_idx, num_tokens, configs.combine);
}

} // namespace deep_gemm::mega
