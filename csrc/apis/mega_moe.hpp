#pragma once

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <limits>
#include <tuple>
#include <vector>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "../jit_kernels/impls/mega_moe.hpp"
#include "../jit_kernels/heuristics/mega_moe.hpp"

namespace deep_gemm::mega_moe {

// Shared API helpers.
namespace py = pybind11;
using Tensor = torch::Tensor;
using Pair = std::pair<Tensor, Tensor>;
using OptTensor = std::optional<Tensor>;
using OptPair = std::optional<Pair>;
using mega_moe_config::DispatchConfig;
using mega_moe_config::TuningConfig;

static void check_device() {
    TORCH_CHECK(is_ppu1v5_device(), "MegaMoE requires PPU1.5");
}

// Host-only state owned by one SymmBuffer. Python never interprets launch
// configurations; each token count and activation retains its cached selection.
struct SymmBufferState {
    DispatchConfig allocation_config;
    std::optional<mega_moe_config::Tile> gemm1_config, gemm2_config;
    std::map<std::pair<int, std::string>, DispatchConfig> dispatch_cache;
    std::map<std::pair<int, std::string>, TuningConfig> tuning_cache;

    const DispatchConfig& dispatch(const int& num_tokens, const TuningConfig& tuning) {
        const auto& c = allocation_config;
        TORCH_CHECK(num_tokens >= 0 && num_tokens <= c.nmtpr);
        const auto key = std::make_pair(num_tokens, tuning.workload.activation);
        auto found = dispatch_cache.find(key);
        if (found == dispatch_cache.end()) {
            auto config = mega_moe_config::fp4_dispatch_config(
                c.num_ranks, c.num_experts, c.num_topk, c.token_bytes * 2, c.gemm1_n,
                c.nmtpr, num_tokens, gemm1_config ? gemm1_config : tuning.gemm1.tile, tuning.gemm1);
            found = dispatch_cache.emplace(key, std::move(config)).first;
        }
        return found->second;
    }

    const TuningConfig& tuning(const int& num_tokens, const std::string& activation) {
        const auto key = std::make_pair(num_tokens, activation);
        auto found = tuning_cache.find(key);
        if (found == tuning_cache.end()) {
            const auto& c = allocation_config;
            auto config = mega_moe_config::tuning_config(
                {c.num_ranks, c.num_experts, c.num_topk, c.token_bytes * 2,
                 c.gemm1_n / 2, num_tokens, activation});
            if (gemm1_config)
                config.gemm1.tile = gemm1_config;
            if (gemm2_config) config.gemm2.tile = gemm2_config;
            if (!config.gemm2.tile) {
                const auto [sms, bm, bn, bk, wm, wn, stages, smem] = mega_moe_config::fp4_scatter_config(
                    c.token_bytes * 2, c.gemm1_n / 4, c.num_experts_per_rank, config);
                config.gemm2.tile = mega_moe_config::Tile{bm, bn, bk, wm, wn, stages};
            }
            found = tuning_cache.emplace(key, std::move(config)).first;
        }
        return found->second;
    }
};

static void check_tensor(const Tensor& tensor, const torch::ScalarType& dtype, const bool& contiguous = true) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == dtype && (!contiguous || tensor.is_contiguous()),
                "Invalid MegaMoE tensor device, dtype or layout");
}
static void check_shape(const Tensor& tensor, c10::IntArrayRef shape) {
    TORCH_CHECK(tensor.sizes() == shape, "Invalid MegaMoE tensor shape: ", tensor.sizes());
}
static void validate_rank(const DispatchConfig& config, const int& rank_idx, const int& num_tokens) {
    check_device();
    TORCH_CHECK(rank_idx >= 0 && rank_idx < config.num_ranks && num_tokens >= 0 && num_tokens <= config.nmtpr);
}

// Internal allocation configuration for the Python symmetric-buffer factory.
static void check_fp4_pipeline_tile(const mega_moe_config::Tile& tile, const int& n, const int& k,
                                    const bool& gemm1) {
    const auto [bm, bn, bk, wm, wn, stages] = tile;
    const char* name = gemm1 ? "gemm1_config" : "gemm2_config";
    TORCH_CHECK(bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256,
                name, " requires BLOCK_M in {16,32,64,128,256}");
    TORCH_CHECK(bn > 0 && (gemm1 ? (bn == 128 || bn == 256) : bn % 256 == 0) && n % bn == 0,
                name, " has an unsupported BLOCK_N for this output width");
    TORCH_CHECK(bk > 0 && bk % 32 == 0 && k % bk == 0,
                name, " requires BLOCK_K divisible by 32 and dividing packed K");
    TORCH_CHECK(wm > 0 && wn > 0 && wm % 16 == 0 && wn % 16 == 0 &&
                bm % wm == 0 && bn % wn == 0,
                name, " has an invalid warp tile");
    const int threads = mega_moe_config::threads(bm, bn, wm, wn);
    TORCH_CHECK(threads >= 32 && threads <= 1024 && stages > 0,
                name, " has an invalid thread or stage count");
    // GEMM1 reduces its requested stage count to fit shared memory. GEMM2
    // uses the requested count directly; reject oversized tiles before allocation.
    const auto smem = mega_moe_config::fp4_smem(gemm1 ? 1 : stages, bm, bn, bk);
    TORCH_CHECK(std::get<0>(smem) <= 256 * 1024, name, " exceeds shared-memory capacity");
}

static void fp4_mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_idx,
                               const Tensor& topk_weights, const Tensor& x, const Tensor& x_sf,
                               const Tensor& staged_topk_idx, const Tensor& staged_topk_weights, const int& num_tokens) {
    check_device();
    check_tensor(hidden_states, torch::kBFloat16);
    check_tensor(topk_idx, torch::kInt64);
    check_tensor(topk_weights, torch::kFloat32);
    check_tensor(x, torch::kUInt8);
    check_tensor(x_sf, torch::kUInt8);
    check_tensor(staged_topk_idx, torch::kInt64);
    check_tensor(staged_topk_weights, torch::kFloat32);
    const int hidden = hidden_states.size(1), num_topk = topk_idx.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 64 == 0 && num_tokens >= 0 && num_tokens <= hidden_states.size(0));
    TORCH_CHECK(x.size(1) == hidden / 2 && x_sf.size(1) == hidden / 32 &&
                staged_topk_idx.size(1) == num_topk && staged_topk_weights.size(1) == num_topk);
    for (const auto& tensor : {x, x_sf, staged_topk_idx, staged_topk_weights, topk_idx, topk_weights})
        TORCH_CHECK(tensor.size(0) >= num_tokens);
    const int threads = mega_moe_config::pre_dispatch_threads(hidden);
    TORCH_CHECK(threads > 0 && threads % 32 == 0 && threads <= hidden / 8);
    mega_moe_native::fp4_mega_moe_pre_dispatch_impl(hidden_states, topk_idx, topk_weights, x,
                                                    x_sf, staged_topk_idx, staged_topk_weights, num_tokens,
                                                    hidden, num_topk, threads);
}

static bool is_fp4_scale_layout(const Tensor& scale) {
    return scale.scalar_type() == torch::kUInt16 &&
           ((scale.dim() == 2 && (scale.stride(0) == 1 || scale.size(0) == 1)) ||
            (scale.dim() == 3 && (scale.stride(1) == 1 || scale.size(1) == 1)));
}
// Symmetric layout and row metadata.
static std::map<std::string, int64_t> symmetric_layout(const DispatchConfig& config) {
    const auto num_ranks = config.num_ranks, num_experts = config.num_experts;
    const auto num_max_tokens_per_rank = config.nmtpr, num_topk = config.num_topk;
    // GEMM row counts and offsets are signed 32-bit. Validate before the
    // workspace constructor computes the device-side pool size in uint32_t.
    const auto pool_rows = layout::get_num_max_pool_tokens<uint64_t>(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks);
    TORCH_CHECK(pool_rows <= std::numeric_limits<int32_t>::max(),
                "MegaMoE token pool exceeds the signed 32-bit row index range");
    const uint64_t token_bytes = config.token_bytes, scale_bytes = config.sf_bytes;
    const layout::Workspace workspace(nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk);
    const auto align16 = [](uint64_t bytes) { return (bytes + 15) / 16 * 16; };
    // Keep source and landing token rows aligned independently of metadata size.
    const auto align128 = [](uint64_t bytes) { return (bytes + 127) / 128 * 128; };
    const uint64_t ws_bytes = workspace.get_num_bytes();
    const uint64_t num_pool_tokens = workspace.num_max_pool_tokens;
    const uint64_t off_it = align128(ws_bytes);
    const uint64_t off_isf = align16(off_it + uint64_t(num_max_tokens_per_rank) * token_bytes);
    const uint64_t off_iti = align16(off_isf + uint64_t(num_max_tokens_per_rank) * scale_bytes);
    const uint64_t off_itw = align16(off_iti + uint64_t(num_max_tokens_per_rank) * num_topk * sizeof(int64_t));
    const uint64_t off_pt = align128(off_itw + uint64_t(num_max_tokens_per_rank) * num_topk * sizeof(float));
    const uint64_t off_psf = align16(off_pt + num_pool_tokens * token_bytes);
    const uint64_t off_pw = align16(off_psf + num_pool_tokens * scale_bytes);
    // W4 routing weights stay on the source rank for combine.
    const uint64_t off_meta = off_pw;
    const uint64_t off_packed = off_meta + layout::Gemm2Metadata::rows_bytes(num_experts / num_ranks);
    std::map<std::string, int64_t> output{
        {"total", off_packed + num_pool_tokens * sizeof(int64_t)},
        {"off_gemm2_meta", off_meta},
        {"off_gemm2_packed", off_packed},
        {"off_it", off_it},
        {"off_isf", off_isf},
        {"off_iti", off_iti},
        {"off_itw", off_itw},
        {"num_pool", num_pool_tokens},
        {"num_pool_blocks", workspace.num_max_pool_blocks},
        {"woff_sc", reinterpret_cast<uintptr_t>(workspace.get_expert_send_count_ptr(0))},
        {"woff_rc", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_ptr(0, 0))},
        {"woff_rcs", reinterpret_cast<uintptr_t>(workspace.get_expert_recv_count_sum_ptr(0))},
        {"woff_meta", reinterpret_cast<uintptr_t>(workspace.get_token_src_metadata_ptr(0))},
    };
    output["woff_arr"] = reinterpret_cast<uintptr_t>(workspace.get_l1_arrival_count_ptr(0));
    return output;
}

// Match upstream's allocation protocol: return bytes and a callable that
// creates persistent views of the caller-allocated symmetric storage.
static std::tuple<int64_t, std::function<py::tuple(const Tensor&)>>
get_symm_buffer_size_for_mega_moe(
    const int& num_ranks, const int& num_experts, const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const std::string& mma_type,
    const std::string& activation,
    const std::optional<mega_moe_config::Tile>& gemm1_config,
    const std::optional<mega_moe_config::Tile>& gemm2_config) {
    check_device();
    TORCH_CHECK(mma_type == "fp4xfp4", "PPU MegaMoE requires mma_type='fp4xfp4' (MXFP4)");
    TORCH_CHECK(activation == "swiglu" || activation == "situ" || activation == "step",
                "unsupported precision/activation combination for W4A4: ", activation);
    TORCH_CHECK(num_ranks > 0 && num_ranks <= 32 && num_experts > 0 && num_experts % num_ranks == 0 &&
                num_topk > 0 && num_topk <= 16 && num_topk <= num_experts &&
                num_max_tokens_per_rank > 0 && num_max_tokens_per_rank < (1 << 24),
                "Invalid MegaMoE ranks, expert count, top-k or token capacity");
    TORCH_CHECK(intermediate_hidden > 0 && intermediate_hidden % 128 == 0,
                "intermediate_hidden must be the activated width I, a positive multiple of 128");
    TORCH_CHECK(hidden > 0 && hidden % 512 == 0, "MegaMoE hidden must be divisible by 512 for W4A4");
    if (gemm1_config)
        check_fp4_pipeline_tile(*gemm1_config, 2 * intermediate_hidden, hidden / 2, true);
    if (gemm2_config)
        check_fp4_pipeline_tile(*gemm2_config, hidden, intermediate_hidden / 2, false);
    auto config = mega_moe_config::fp4_dispatch_config(
        num_ranks, num_experts, num_topk, hidden, 2 * intermediate_hidden,
        num_max_tokens_per_rank, num_max_tokens_per_rank, gemm1_config);
    auto state = std::make_shared<SymmBufferState>(SymmBufferState{
        config, gemm1_config, gemm2_config, {}, {}});
    const auto offsets = symmetric_layout(config);
    const int64_t num_bytes = offsets.at("total");
    auto slice_input_buffers = [=](const Tensor& buffer) {
        check_tensor(buffer, torch::kUInt8);
        TORCH_CHECK(buffer.dim() == 1 && buffer.nbytes() >= static_cast<uint64_t>(num_bytes),
                    "symmetric buffer is smaller than the MegaMoE workspace");
        const auto slice = [&](const int64_t& offset, const std::vector<int64_t>& shape, const torch::ScalarType& dtype) {
            int64_t elements = 1;
            for (const auto size : shape) elements *= size;
            return buffer.narrow(0, offset, elements * c10::elementSize(dtype)).view(dtype).view(shape);
        };
        const int64_t pool = offsets.at("num_pool"), groups = num_experts / num_ranks;
        const auto rows = slice(offsets.at("off_gemm2_meta"), {5, groups}, torch::kInt32);
        const int64_t counts_begin = offsets.at("woff_sc");
        const int64_t counts_end = offsets.at("woff_rcs") + groups * sizeof(int64_t);
        return py::make_tuple(
            slice(offsets.at("off_it"), {num_max_tokens_per_rank, hidden / 2}, torch::kUInt8),
            slice(offsets.at("off_isf"), {num_max_tokens_per_rank, hidden / 32}, torch::kUInt8),
            slice(offsets.at("off_iti"), {num_max_tokens_per_rank, num_topk}, torch::kInt64),
            slice(offsets.at("off_itw"), {num_max_tokens_per_rank, num_topk}, torch::kFloat32),
            slice(offsets.at("woff_rc"), {num_experts}, torch::kInt64),
            slice(offsets.at("woff_meta"), {pool, 3}, torch::kInt32),
            slice(offsets.at("woff_rcs"), {groups}, torch::kInt64),
            slice(offsets.at("woff_arr"), {offsets.at("num_pool_blocks")}, torch::kUInt32),
            rows[0], rows[1], rows[2], rows[3], rows[4],
            slice(offsets.at("off_gemm2_packed"), {pool}, torch::kInt64),
            buffer.narrow(0, counts_begin, counts_end - counts_begin), state);
    };
    return {num_bytes, slice_input_buffers};
}

void mega_moe_pre_dispatch(const Tensor& hidden_states, const Tensor& topk_idx,
                           const Tensor& topk_weights, const Tensor& x, const Tensor& x_sf,
                           const Tensor& staged_topk_idx, const Tensor& staged_topk_weights,
                           const std::optional<int>& requested_tokens) {
    check_device();
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [tokens, hidden]");
    TORCH_CHECK(x.dim() == 2 && x_sf.dim() == 2 && staged_topk_idx.dim() == 2 &&
                staged_topk_weights.dim() == 2, "pre-dispatch destinations must be matrices");
    const int num_tokens = requested_tokens.value_or(hidden_states.size(0));
    const int capacity = x.size(0), hidden = x.size(1) * 2, num_topk = staged_topk_idx.size(1);
    TORCH_CHECK(hidden > 0 && hidden % 512 == 0,
                "W4A4 dispatch requires hidden divisible by 512 for scale copies");
    check_shape(x_sf, {capacity, hidden / 32});
    check_shape(staged_topk_idx, {capacity, num_topk});
    check_shape(staged_topk_weights, {capacity, num_topk});
    const auto device = x.device();
    TORCH_CHECK(hidden_states.device() == device && topk_idx.device() == device && topk_weights.device() == device,
                "pre-dispatch inputs must be on the workspace device");
    TORCH_CHECK(x_sf.device() == device && staged_topk_idx.device() == device &&
                staged_topk_weights.device() == device, "pre-dispatch destinations must be on the workspace device");
    TORCH_CHECK(topk_idx.dim() == 2 && topk_weights.dim() == 2,
                "topk_ids and topk_weights must have shape [tokens, topk]");
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= capacity,
                "pre-dispatch token count must be within workspace capacity");
    TORCH_CHECK(hidden_states.size(0) >= num_tokens && topk_idx.size(0) >= num_tokens &&
                topk_weights.sizes() == topk_idx.sizes(),
                "pre-dispatch inputs must contain num_tokens rows and matching routing shapes");
    TORCH_CHECK(hidden_states.size(1) == hidden && topk_idx.size(1) == num_topk,
                "pre-dispatch hidden and topk dimensions must match the workspace");
    const auto input = hidden_states.narrow(0, 0, num_tokens);
    const auto ids = topk_idx.narrow(0, 0, num_tokens), weights = topk_weights.narrow(0, 0, num_tokens);
    fp4_mega_moe_pre_dispatch(input, ids, weights, x, x_sf, staged_topk_idx, staged_topk_weights, num_tokens);
}

static std::string check_activation(const std::string& activation, const std::optional<float>& activation_clamp,
                                       const float& situ_beta, const float& situ_linear_beta) {
    TORCH_CHECK(activation == "swiglu" || activation == "step" || activation == "situ",
                "unsupported precision/activation combination for W4A4: ", activation);
    TORCH_CHECK(!activation_clamp || (std::isfinite(*activation_clamp) && *activation_clamp > 0 && activation != "situ"),
                "activation_clamp requires SwiGLU or Step and a finite positive limit");
    TORCH_CHECK(activation == "situ" || (situ_beta == 4.0f && situ_linear_beta == 25.0f),
                "SiTU parameters require activation='situ'");
    TORCH_CHECK(activation != "situ" || (std::isfinite(situ_beta) && situ_beta > 0 &&
                                  std::isfinite(situ_linear_beta) && situ_linear_beta > 0),
                "situ_beta and situ_linear_beta must be finite and positive");
    return activation == "swiglu" ? "silu_and_mul" : activation == "step" ? "silu_and_mul_step" : "situ_and_mul";
}

// Validate both stages before entering a cross-rank kernel. A malformed GEMM2
// weight must not leave peers waiting after GEMM1 has already advanced the protocol.
static void check_pipeline_weights(const Pair& l1_weights, const Pair& l2_weights,
                                   const OptPair& l1_weights_replicas, const OptPair& l2_weights_replicas,
                                   const int& num_experts_per_rank, const int& hidden, const int& intermediate_hidden,
                                   const torch::Device& device) {
    TORCH_CHECK(l1_weights_replicas.has_value() == l2_weights_replicas.has_value(),
                "both stages must use the same master/replica partition");
    const auto check = [&](const Pair& weights, const int& n, const int& k, const bool& is_l1) {
        check_tensor(weights.first, torch::kUInt8);
        TORCH_CHECK(weights.first.dim() == 3 && weights.first.size(0) > 0, "weights must contain local experts");
        const int num_weight_experts = weights.first.size(0);
        check_shape(weights.first, {num_weight_experts, n, k / 2});
        TORCH_CHECK(weights.first.device() == device && weights.second.device() == device,
                    "weights must be on the workspace device");
        check_tensor(weights.second, torch::kUInt16, is_l1);
        if (is_l1) check_shape(weights.second, {num_weight_experts, n * k / 64});
        else {
            check_shape(weights.second, {num_weight_experts, n, k / 64});
            TORCH_CHECK(is_fp4_scale_layout(weights.second), "invalid transformed W4A4 GEMM2 scales");
        }
    };
    check(l1_weights, 2 * intermediate_hidden, hidden, true);
    check(l2_weights, hidden, intermediate_hidden, false);
    TORCH_CHECK(l1_weights.first.size(0) == l2_weights.first.size(0), "GEMM1/GEMM2 master counts must match");
    if (l1_weights_replicas) {
        check(*l1_weights_replicas, 2 * intermediate_hidden, hidden, true);
        check(*l2_weights_replicas, hidden, intermediate_hidden, false);
        TORCH_CHECK(l1_weights_replicas->first.size(0) == l2_weights_replicas->first.size(0),
                    "GEMM1/GEMM2 replica counts must match");
    }
    TORCH_CHECK(l1_weights.first.size(0) + (l1_weights_replicas ? l1_weights_replicas->first.size(0) : 0) == num_experts_per_rank,
                "master and replica counts must match the workspace's local physical experts");
}

void fp4_fp4_mega_moe(
    const Tensor& y, const Pair& l1_weights, const Pair& l2_weights,
    const OptTensor& cumulative_local_expert_recv_stats, const Tensor& sym_buffer,
    const Tensor& sym_buffer_ptrs, const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk, const std::tuple<int, int, int>& recipe,
    const std::string& activation, const std::optional<float>& activation_clamp, const bool& fast_math,
    const std::shared_ptr<SymmBufferState>& state,
    const Tensor& l2_acts, const Tensor& l2_acts_sf, const Tensor& sfa_scratch,
    const Tensor& m_rows, const Tensor& valid_m_rows, const Tensor& src_m_offsets, const Tensor& token_meta,
    const OptTensor& block_info, const Tensor& combine, const Tensor& combine_offsets,
    const Tensor& dispatch_counts, const Tensor& topk_weights,
    const Tensor& barrier_pointers, const Tensor& barrier_done,
    const OptPair& l1_weights_replicas, const OptPair& l2_weights_replicas,
    const float& situ_beta, const float& situ_linear_beta) {
    TORCH_CHECK(state, "MegaMoE workspace configuration is missing");
    check_device();
    check_tensor(sym_buffer, torch::kUInt8);
    TORCH_CHECK(y.dim() == 2, "output must be [tokens, hidden]");
    const auto& allocation = state->allocation_config;
    const int num_tokens = y.size(0), hidden = y.size(1), num_ranks = allocation.num_ranks,
              intermediate_hidden = allocation.gemm1_n / 2;
    TORCH_CHECK(num_experts == allocation.num_experts && num_topk == allocation.num_topk &&
                hidden == allocation.token_bytes * 2 && num_max_tokens_per_rank == allocation.nmtpr,
                "MegaMoE workspace dimensions must not change after allocation");
    const auto transport = mega_moe_native::make_sym_buffer(sym_buffer_ptrs, rank_idx, num_ranks);
    TORCH_CHECK(transport.base == reinterpret_cast<int64_t>(sym_buffer.data_ptr()),
                "symmetric pointer table must reference the local workspace buffer");
    const auto device = sym_buffer.device();
    check_tensor(y, torch::kBFloat16);
    check_shape(y, {num_tokens, hidden});
    TORCH_CHECK(num_tokens <= num_max_tokens_per_rank, "output token count exceeds workspace capacity");
    TORCH_CHECK(y.device() == device, "output must be on the workspace device");
    TORCH_CHECK(recipe == std::make_tuple(1, 1, 32),
                "PPU MXFP4 MegaMoE requires recipe=(1, 1, 32)");
    if (cumulative_local_expert_recv_stats) {
        const auto& stats = *cumulative_local_expert_recv_stats;
        check_tensor(stats, torch::kInt32);
        TORCH_CHECK(stats.device() == device && stats.numel() == num_experts / num_ranks,
                    "cumulative_local_expert_recv_stats must have one INT32 element per local physical expert "
                    "on the workspace device");
    }
    const auto activation_op = check_activation(activation, activation_clamp, situ_beta, situ_linear_beta);
    check_pipeline_weights(l1_weights, l2_weights, l1_weights_replicas, l2_weights_replicas,
                           num_experts / num_ranks, hidden, intermediate_hidden, device);
    const auto& tuning = state->tuning(num_tokens, activation);
    check_tensor(l2_acts, torch::kUInt8);
    check_tensor(l2_acts_sf, torch::kUInt16, false);
    TORCH_CHECK(l2_acts.dim() == 2, "W4A4 GEMM2 activations must be a matrix");
    const int num_pool_tokens = l2_acts.size(0);
    check_shape(l2_acts, {num_pool_tokens, intermediate_hidden / 2});
    check_shape(l2_acts_sf, {num_pool_tokens, intermediate_hidden / 64});
    TORCH_CHECK(l2_acts_sf.stride(0) == 1 && l2_acts_sf.stride(1) == num_pool_tokens,
                "W4A4 activation scales must be column-major");
    const auto& dispatch = state->dispatch(num_tokens, tuning);
    validate_rank(dispatch, rank_idx, num_tokens);
    const mega_moe_native::MegaMoEWorkspace workspace{
        dispatch, 256,
        sym_buffer_ptrs, l2_acts, l2_acts_sf,
        sfa_scratch.narrow(0, 0, int64_t(num_pool_tokens) * (dispatch.sf_bytes / 2)),
        m_rows, valid_m_rows, src_m_offsets, token_meta, block_info,
        combine, combine_offsets, dispatch_counts, topk_weights, barrier_pointers, barrier_done};
    mega_moe_native::fp4_fp4_mega_moe_impl(y, l1_weights, l2_weights, workspace, rank_idx, num_tokens, activation_op,
        activation_clamp.value_or(0.0f), fast_math, situ_beta, situ_linear_beta,
        l1_weights_replicas, l2_weights_replicas, tuning, cumulative_local_expert_recv_stats);
}

// Python bindings.
void register_apis(py::module_& m) {
    py::class_<SymmBufferState, std::shared_ptr<SymmBufferState>>(m, "_MegaMoESymmBufferState");
    m.def("fp4_fp4_mega_moe", &fp4_fp4_mega_moe);
    m.def("mega_moe_pre_dispatch", &mega_moe_pre_dispatch);
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe);
}

} // namespace deep_gemm::mega_moe
