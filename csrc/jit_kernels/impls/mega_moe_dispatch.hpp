#pragma once

#include <torch/extension.h>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "mega_moe_runtime.hpp"
#include "../heuristics/mega_moe.hpp"

namespace deep_gemm::mega_moe_native {

static_assert(sizeof(layout::SymBuffer<1>) == 592, "MegaMoE symmetric buffer ABI changed");

using mega_moe_config::DispatchConfig;
using mega_moe_config::PreDispatchConfig;
using mega_moe_config::ScatterConfig;
using SourceFields = std::vector<std::pair<std::string, std::string>>;

inline std::string source_float_literal(float value) {
    TORCH_CHECK(std::isfinite(value), "MegaMoE specialization must be finite");
    return fmt::format("{:.9e}f", value);
}

inline void replace_source_field(std::string& code, const std::string& key, const std::string& text) {
    const auto marker = "{" + key + "}";
    for (size_t pos = code.find(marker); pos != std::string::npos; pos = code.find(marker, pos + text.size()))
        code.replace(pos, marker.size(), text);
}

inline SourceFields source_fields(const DispatchConfig& config) {
    SourceFields values{
        {"GEMM1_DYNAMIC_QUEUE", config.gemm1_dynamic_queue ? "true" : "false"},
        {"BLOCK_M", std::to_string(config.block_m)},
        {"CLAMP_VALUE", source_float_literal(config.activation_clamp)},
        {"FAST_MATH", config.fast_math ? "true" : "false"},
        {"FUSED_SITU", (config.fused_situ ? "true" : "false")},
        {"FUSED_SWIGLU", (config.fused_swiglu ? "true" : "false")},
        {"GEMM2_BLOCK_M", std::to_string(config.gemm2_block_m)},
        {"GEMM1_BLOCK_K", std::to_string(config.gemm1_block_k)},
        {"GEMM1_BLOCK_N", std::to_string(config.gemm1_block_n)},
        {"GEMM1_N", std::to_string(config.gemm1_n)},
        {"GEMM1_STAGES", std::to_string(config.gemm1_stages)},
        {"GEMM1_WARP_M", std::to_string(config.gemm1_warp_m)},
        {"GEMM1_WARP_N", std::to_string(config.gemm1_warp_n)},
        {"GEMM_SMS", std::to_string(config.gemm_sms)},
        {"NMTPR", std::to_string(config.nmtpr)},
        {"NUM_EXPERTS", std::to_string(config.num_experts)},
        {"NUM_EXPERTS_PER_RANK", std::to_string(config.num_experts_per_rank)},
        {"NUM_RANKS", std::to_string(config.num_ranks)},
        {"NUM_SMS", std::to_string(config.num_sms)},
        {"NUM_THREADS", std::to_string(config.num_threads)},
        {"NUM_TOPK", std::to_string(config.num_topk)},
        {"PULL_SLOTS", std::to_string(config.pull_slots)},
        {"PULL_SMS", std::to_string(config.pull_sms)},
        {"SFB_LAYOUT_BLOCK_N", std::to_string(config.sfb_layout_block_n)},
        {"SF_BYTES", std::to_string(config.sf_bytes)},
        {"SILU_THEN_CLAMP", (config.silu_then_clamp ? "true" : "false")},
        {"SITU_BETA", source_float_literal(config.situ_beta)},
        {"SITU_LINEAR_BETA", source_float_literal(config.situ_linear_beta)},
        {"TOKEN_BYTES", std::to_string(config.token_bytes)},
    };
    TORCH_CHECK(config.num_ranks > 0 && config.num_experts % config.num_ranks == 0, "invalid MegaMoE layout");
    const layout::Workspace workspace(nullptr, config.num_ranks, config.num_experts, config.nmtpr, config.num_topk);
    const auto align16 = [](uint64_t bytes) { return (bytes + 15) / 16 * 16; };
    // Match the source and landing token alignment used by symmetric_layout().
    const auto align128 = [](uint64_t bytes) { return (bytes + 127) / 128 * 128; };
    const uint64_t pool = workspace.num_max_pool_tokens, capacity = config.nmtpr;
    const uint64_t it = align128(workspace.get_num_bytes());
    const uint64_t isf = align16(it + capacity * config.token_bytes);
    const uint64_t iti = align16(isf + capacity * config.sf_bytes);
    const uint64_t itw = align16(iti + capacity * config.num_topk * sizeof(int64_t));
    const uint64_t pt = align128(itw + capacity * config.num_topk * sizeof(float));
    const uint64_t psf = align16(pt + pool * config.token_bytes);
    const uint64_t pw = align16(psf + pool * config.sf_bytes);
    for (const auto& [key, value] : std::vector<std::pair<const char*, uint64_t>>{
             {"LAYOUT_WS_BYTES", workspace.get_num_bytes()},
             {"LAYOUT_NUM_POOL", pool},
             {"LAYOUT_NUM_POOL_BLOCKS", workspace.num_max_pool_blocks},
             {"LAYOUT_OFF_IT", it},
             {"LAYOUT_OFF_ISF", isf},
             {"LAYOUT_OFF_ITI", iti},
             {"LAYOUT_OFF_ITW", itw},
             {"LAYOUT_OFF_PT", pt},
             {"LAYOUT_OFF_PSF", psf},
             {"LAYOUT_OFF_PW", pw},
             {"LAYOUT_OFF_W4_GEMM2_META", pw},
             {"LAYOUT_WOFF_RCS", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_sum_ptr(0))}})
        values.emplace_back(key, std::to_string(value));
    return values;
}

inline SourceFields source_fields(const PreDispatchConfig& config) {
    SourceFields values{
        {"BLOCK_THREADS", std::to_string(config.block_threads)},
        {"HIDDEN", std::to_string(config.hidden)},
        {"NUM_SMS", std::to_string(config.num_sms)},
        {"TOPK", std::to_string(config.num_topk)},
        {"TOP_K", std::to_string(config.num_topk)},
    };
    return values;
}

template <typename Runtime, typename Config>
inline std::string rendered_source(const std::string& name, std::string source, const Config& config) {
    const auto fields = source_fields(config);
    std::string key = name;
    for (const auto& [field, value] : fields)
        key += ":" + field + "=" + value;
    static std::mutex mutex;
    static std::unordered_map<std::string, std::string> sources;
    const std::lock_guard<std::mutex> guard(mutex);
    const auto found = sources.find(key);
    if (found != sources.end())
        return found->second;
    const std::string marker = "PPU_KERNEL_ENTRY";
    for (size_t pos = source.find(marker); pos != std::string::npos;
         pos = source.find(marker, pos + name.size()))
        source.replace(pos, marker.size(), name);
    for (const auto& [field, value] : fields)
        replace_source_field(source, field, value);
    return sources.emplace(key, std::move(source)).first->second;
}

inline void* device_pointer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "MegaMoE expects CUDA device tensors");
    return tensor.data_ptr();
}

template <typename T> inline T* device_pointer(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(device_pointer(tensor));
}

inline layout::SymBuffer<1> make_sym_buffer(const torch::Tensor& pointers, int rank_idx, int num_ranks) {
    TORCH_CHECK(!pointers.is_cuda() && pointers.scalar_type() == torch::kInt64 && pointers.is_contiguous() &&
                    pointers.numel() == num_ranks && num_ranks > 0 && num_ranks <= layout::kNumMaxRanks && rank_idx >= 0 &&
                    rank_idx < num_ranks,
                "invalid CPU symmetric pointer table");
    layout::SymBuffer<1> sym_buffer{};
    sym_buffer.base = pointers.data_ptr<int64_t>()[rank_idx];
    sym_buffer.rank_idx = rank_idx;
    for (int peer = 0; peer < num_ranks; ++peer)
        sym_buffer.offsets[peer] = pointers.data_ptr<int64_t>()[peer] - sym_buffer.base;
    return sym_buffer;
}

struct FP4MegaMoELinear1Params {
    layout::SymBuffer<1> sym_buffer;
    int rank_idx, num_tokens;
    uint8_t* l1_weights;
    uint16_t* l1_weights_sf;
    uint8_t* l1_weights_replicas;
    uint16_t* l1_weights_replicas_sf;
    int num_master_experts;
    uint16_t* sfa_scratch;
    uint8_t* l2_acts;
    uint16_t* l2_acts_sf;
    int32_t* cumulative_local_expert_recv_stats;
};

struct FP4MegaMoELinear1Args {
    DispatchConfig launch_info;
    LaunchArgs launch_args;
    FP4MegaMoELinear1Params kernel_params;
};

inline std::string generate_fp4_mega_moe_linear1(const FP4MegaMoELinear1Args& args) {
    return rendered_source<FP4MegaMoELinear1Args>("fp4_mega_moe_dispatch_gemm1", R"DGMEGA(#include <deep_gemm/impls/fp4_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
static_assert(sizeof(layout::SymBuffer<{NUM_RANKS}>) == 592, "MegaMoE symmetric buffer ABI changed");
constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};
constexpr uint32_t ppu_smem_compute() {
constexpr int stages = mega_moe::fp4::Gemm1PickStage<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, GEMM1_STAGES, 262144, {SFB_LAYOUT_BLOCK_N}>::value;
using Cfg = mega_moe::fp4::Gemm1MmaConfig<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    GEMM1_WARP_M, GEMM1_WARP_N, stages, {SFB_LAYOUT_BLOCK_N}>;
uint32_t collective = sizeof(typename Cfg::CollectiveMainloop::SharedStorage);
uint32_t phase_a = NUM_EXPERTS * sizeof(uint32_t);
uint32_t pull = {PULL_SLOTS} * (TOKEN_BYTES + SF_BYTES);
return collective > phase_a ? (collective > pull ? collective : pull) : (phase_a > pull ? phase_a : pull);
}
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = ppu_smem_compute();
extern "C" __global__ __launch_bounds__(NUM_THREADS, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<{NUM_RANKS}> sb, int rank, int num_tokens, uint8_t* gemm1_weight, uint16_t* gemm1_weight_sf_u16, uint8_t* gemm1_weight_replicas, uint16_t* gemm1_weight_sf_u16_replicas, int num_master_experts, uint16_t* sfa_scratch, uint8_t* fp4_output, uint16_t* scale_output, int32_t* cumulative_local_expert_recv_stats) {

using namespace deep_gemm;

constexpr uint32_t NUM_RANKS = {NUM_RANKS};
constexpr uint32_t NUM_SMS = {NUM_SMS};
constexpr uint32_t NUM_THREADS = {NUM_THREADS};
constexpr uint32_t NUM_EXPERTS = {NUM_EXPERTS};
constexpr uint32_t NUM_EXPERTS_PER_RANK = {NUM_EXPERTS_PER_RANK};
constexpr uint32_t NUM_TOPK = {NUM_TOPK};
constexpr uint32_t BLOCK_M = {BLOCK_M};
constexpr uint32_t NMTPR = {NMTPR};
constexpr uint32_t TOKEN_BYTES = {TOKEN_BYTES};
constexpr uint32_t SF_BYTES = {SF_BYTES};
constexpr uint32_t GEMM1_N = {GEMM1_N};
constexpr uint32_t GEMM1_BLOCK_N = {GEMM1_BLOCK_N};
constexpr uint32_t GEMM1_BLOCK_K = {GEMM1_BLOCK_K};
constexpr uint32_t GEMM1_WARP_M = {GEMM1_WARP_M};
constexpr uint32_t GEMM1_WARP_N = {GEMM1_WARP_N};
constexpr uint32_t GEMM1_STAGES = {GEMM1_STAGES};

constexpr uint64_t ws_bytes = {LAYOUT_WS_BYTES};
constexpr uint64_t num_pool = {LAYOUT_NUM_POOL};
constexpr uint64_t num_pool_blocks = {LAYOUT_NUM_POOL_BLOCKS};
constexpr uint64_t off_ws = 0, off_it = {LAYOUT_OFF_IT}, off_isf = {LAYOUT_OFF_ISF};
constexpr uint64_t off_iti = {LAYOUT_OFF_ITI};
constexpr uint64_t off_pt = {LAYOUT_OFF_PT}, off_psf = {LAYOUT_OFF_PSF};


const int64_t local_base = sb.base;
auto P = [&](uint64_t off) -> void* { return (void*)(local_base + (int64_t)off); };
void* ws_base          = P(off_ws);
const int64_t* itopk   = (const int64_t*)P(off_iti);
const uint8_t* itok    = (const uint8_t*)P(off_it);
const uint8_t* isf     = (const uint8_t*)P(off_isf);
uint8_t* ptok          = (uint8_t*)P(off_pt);
uint8_t* psf           = (uint8_t*)P(off_psf);
void* gemm2_metadata  = P({LAYOUT_OFF_W4_GEMM2_META});

const uint8_t*  gw     = gemm1_weight;
const uint16_t* gwsf16 = gemm1_weight_sf_u16;
const uint8_t*  gwr    = gemm1_weight_replicas;
const uint16_t* gwsf16r = gemm1_weight_sf_u16_replicas;

void* fp4_output_ptr = (void*)fp4_output;
void* scale_output_ptr = (void*)scale_output;

constexpr int kGemm1Stages = mega_moe::fp4::Gemm1PickStage<
    BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K, GEMM1_WARP_M, GEMM1_WARP_N,
    (int)GEMM1_STAGES, 262144, {SFB_LAYOUT_BLOCK_N}>::value;
static_assert(kGemm1Stages >= 1, "gemm1 needs at least 1 pipeline stage");

using Cfg = mega_moe::fp4::Gemm1MmaConfig<BLOCK_M, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
                                GEMM1_WARP_M, GEMM1_WARP_N, kGemm1Stages,
                                {SFB_LAYOUT_BLOCK_N}>;
uint32_t smem_collective = (uint32_t)sizeof(typename Cfg::CollectiveMainloop::SharedStorage);
uint32_t smem_phaseA = NUM_EXPERTS * sizeof(uint32_t);
uint32_t smem = smem_collective > smem_phaseA ? smem_collective : smem_phaseA;
uint32_t smem_pull = (uint32_t){PULL_SLOTS} * (TOKEN_BYTES + SF_BYTES);
smem = smem > smem_pull ? smem : smem_pull;

float activation_clamp = {CLAMP_VALUE};
bool fused_swiglu = {FUSED_SWIGLU};
bool fused_situ = {FUSED_SITU};
float situ_beta = {SITU_BETA};
float situ_linear_beta = {SITU_LINEAR_BETA};
uint8_t* fp4_out = (uint8_t*)fp4_output_ptr;
uint8_t* scale_out = (uint8_t*)scale_output_ptr;

deep_gemm::mega_moe::fp4::dispatch_gemm1_device<
    NUM_RANKS, NUM_SMS, NUM_THREADS, NUM_EXPERTS,
    NUM_EXPERTS_PER_RANK, NUM_TOPK, BLOCK_M,
    {PULL_SMS},
    {GEMM_SMS},
    {PULL_SLOTS},
    GEMM1_N, TOKEN_BYTES, GEMM1_BLOCK_N, GEMM1_BLOCK_K,
    {SFB_LAYOUT_BLOCK_N},
    GEMM1_WARP_M, GEMM1_WARP_N, kGemm1Stages,
    {FUSED_SITU},
    {SILU_THEN_CLAMP}, {GEMM2_BLOCK_M}, {GEMM1_DYNAMIC_QUEUE}, {FAST_MATH}>(
    ws_base, sb, itopk, itok, isf, ptok, psf, gemm2_metadata, cumulative_local_expert_recv_stats,
    (uint32_t)num_tokens, NMTPR, TOKEN_BYTES, SF_BYTES,
    gw, gwsf16,
    gwr, gwsf16r, (uint32_t)num_master_experts, sfa_scratch,
    activation_clamp, fused_swiglu, fused_situ,
    situ_beta, situ_linear_beta,
    fp4_out, scale_out, (int)rank);

}
)DGMEGA", args.launch_info);
}

inline FP4MegaMoELinear1Args make_fp4_mega_moe_linear1_args(
    const DispatchConfig& config, const torch::Tensor& sym_buffer_ptrs, int rank_idx, int num_tokens,
    const torch::Tensor& l1_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l1_weights_replicas,
    const torch::Tensor& l1_weights_replicas_sf, int num_master_experts,
    const torch::Tensor& sfa_scratch,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats) {
    return FP4MegaMoELinear1Args{
        config, {dim3(config.num_sms), dim3(config.num_threads), 0},
        {make_sym_buffer(sym_buffer_ptrs, rank_idx, config.num_ranks), rank_idx, num_tokens,
         device_pointer<uint8_t>(l1_weights), device_pointer<uint16_t>(l1_weights_sf),
         device_pointer<uint8_t>(l1_weights_replicas), device_pointer<uint16_t>(l1_weights_replicas_sf),
         num_master_experts, device_pointer<uint16_t>(sfa_scratch),
         device_pointer<uint8_t>(l2_acts), device_pointer<uint16_t>(l2_acts_sf),
         cumulative_local_expert_recv_stats ? device_pointer<int32_t>(*cumulative_local_expert_recv_stats) : nullptr}};
}

class FP4MegaMoEPreDispatchRuntime final : public LaunchRuntime<FP4MegaMoEPreDispatchRuntime> {
public:
    struct KernelParams {
        at::BFloat16* hidden_states;
        int64_t* topk_idx;
        float* topk_weights;
        uint8_t* x;
        uint8_t* x_sf;
        int64_t* staged_topk_idx;
        float* staged_topk_weights;
        int num_tokens;
    };

    struct Args {
        PreDispatchConfig launch_info;
        LaunchArgs launch_args;
        KernelParams kernel_params;
    };

    static std::string generate_impl(const Args& args) {
        return rendered_source<FP4MegaMoEPreDispatchRuntime>("fp4_mega_moe_pre_dispatch", R"DGMEGA(#include <deep_gemm/impls/fp4_mega_moe_pre_dispatch.cuh>
using namespace deep_gemm;
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = 0;
extern "C" __global__ __launch_bounds__({BLOCK_THREADS}, 4)
void PPU_KERNEL_ENTRY(__ppu_bfloat16* hidden_states, int64_t* topk_ids, float* topk_weights, uint8_t* input_token, uint8_t* input_sf, int64_t* input_topk_idx, float* input_topk_weights, int num_tokens) {
mega_moe::fp4::fp4_mega_moe_pre_dispatch_kernel_device<{HIDDEN}, {TOP_K}, {BLOCK_THREADS}>(hidden_states,topk_ids,topk_weights,input_token,input_sf,input_topk_idx,input_topk_weights,(uint32_t)num_tokens);
}
)DGMEGA", args.launch_info);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        auto& p = args.kernel_params;
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, p.hidden_states, p.topk_idx, p.topk_weights,
            p.x, p.x_sf, p.staged_topk_idx, p.staged_topk_weights, p.num_tokens));
    }
};

inline void fp4_mega_moe_pre_dispatch_kernel(
    const PreDispatchConfig& config, const torch::Tensor& hidden_states, const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights, const torch::Tensor& x, const torch::Tensor& x_sf,
    const torch::Tensor& staged_topk_idx, const torch::Tensor& staged_topk_weights, int num_tokens) {
    using Runtime = FP4MegaMoEPreDispatchRuntime;
    auto args = Runtime::Args{
        config, {dim3(std::min(num_tokens, config.num_sms * 4)), dim3(config.block_threads), 0},
        {device_pointer<at::BFloat16>(hidden_states), device_pointer<int64_t>(topk_idx), device_pointer<float>(topk_weights),
         device_pointer<uint8_t>(x), device_pointer<uint8_t>(x_sf), device_pointer<int64_t>(staged_topk_idx),
         device_pointer<float>(staged_topk_weights), num_tokens}};
    const auto code = Runtime::generate(args);
    const auto runtime = compiler->build("fp4_mega_moe_pre_dispatch", code);
    args.launch_args.smem_size = get_kernel_smem_size(runtime);
    TORCH_CHECK(get_kernel_occupancy(runtime, config.block_threads, args.launch_args.smem_size) >= 1,
                "MegaMoE launch cannot keep a block resident");
    Runtime::launch(runtime, args);
}

inline void fp4_mega_moe_pre_dispatch_impl(const torch::Tensor& hidden_states, const torch::Tensor& topk_idx,
                                           const torch::Tensor& topk_weights,
                                           const torch::Tensor& x, const torch::Tensor& x_sf,
                                           const torch::Tensor& staged_topk_idx,
                                           const torch::Tensor& staged_topk_weights, int num_tokens,
                                           int hidden, int num_topk, int threads) {
    if (num_tokens == 0)
        return;
    PreDispatchConfig config;
    config.hidden = hidden;
    config.num_topk = num_topk;
    config.block_threads = threads;
    config.num_sms = get_num_sms();
    mega_moe_native::fp4_mega_moe_pre_dispatch_kernel(config, hidden_states, topk_idx, topk_weights,
                                                      x, x_sf, staged_topk_idx,
                                                      staged_topk_weights, num_tokens);
}

} // namespace deep_gemm::mega_moe_native
