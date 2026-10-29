#pragma once
#include "mega_moe.hpp"
#include "m_grouped_w4a8_gemm.hpp"

namespace deep_gemm::mega_moe_native {

struct W4A8DispatchTag {};
inline void w4a8_dispatch(DispatchConfig c, const torch::Tensor& pointers, int rank, int tokens,
    const torch::Tensor& w, const torch::Tensor& s, float inv,
    const torch::Tensor& rw, const torch::Tensor& rs, float rinv,
    const torch::Tensor& output, float limit) {
    const std::string name = "w4a8_mega_moe_dispatch_gemm1";
    const std::string source = R"(
#include <deep_gemm/impls/w4a8_mega_moe.cuh>
#include <deep_gemm/impls/fp8_channelwise_mega_moe_dispatch_gemm1.cuh>
using namespace deep_gemm;
using Backend = mega_moe::w4a8::Gemm1<{BLOCK_M}, {GEMM1_BLOCK_N}, {TOKEN_BYTES},
                                    {GEMM1_WARP_M}, {GEMM1_STAGES}, {SWIGLU_OAI}, {FUSED_SITU}>;
constexpr uint32_t compute_bytes = sizeof(typename Backend::Kernel::SharedStorage);
constexpr uint32_t dispatch_bytes = (({NUM_EXPERTS} * 4 + 15) & ~15u) + {NUM_THREADS} / 32 * {TOKEN_BYTES};
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes =
    compute_bytes > dispatch_bytes ? compute_bytes : dispatch_bytes;
extern "C" __global__ __launch_bounds__({NUM_THREADS}, 1)
void PPU_KERNEL_ENTRY(layout::SymBuffer<{NUM_RANKS}> sb, int tokens,
    const uint8_t* w, const uint8_t* s, float inv, const uint8_t* rw, const uint8_t* rs, float rinv,
    int masters, __ppu_bfloat16* out, float limit) {
    auto p = [&](int64_t off) { return reinterpret_cast<uint8_t*>(sb.base + off); };
    mega_moe::fp8_channelwise::dispatch_gemm1_device<
        {NUM_RANKS}, {NUM_SMS}, {NUM_THREADS}, {NUM_EXPERTS}, {NUM_EXPERTS_PER_RANK},
        {NUM_TOPK}, {BLOCK_M}, {TOKEN_BYTES}, {GEMM1_N}, {GEMM1_BLOCK_N},
        128, {GEMM1_WARP_M}, 64, {GEMM1_STAGES}, {NUM_PULL_SMS}, {SWIGLU_OAI}, true, Backend>(
        p(0), sb, reinterpret_cast<int64_t*>(p({LAYOUT_OFF_ITI})),
        reinterpret_cast<float*>(p({LAYOUT_OFF_ITW})), p({LAYOUT_OFF_IT}),
        reinterpret_cast<float*>(p({LAYOUT_OFF_ISF})), p({LAYOUT_OFF_PT}),
        reinterpret_cast<float*>(p({LAYOUT_OFF_PSF})), reinterpret_cast<float*>(p({LAYOUT_OFF_PW})),
        reinterpret_cast<const cutlass::float_e4m3_t*>(w), reinterpret_cast<const float*>(s),
        reinterpret_cast<const cutlass::float_e4m3_t*>(rw), reinterpret_cast<const float*>(rs),
        masters, out, limit, {SWIGLU_ALPHA}, tokens, {NMTPR}, inv, rinv, {SITU_BETA}, {SITU_LINEAR_BETA});
}
)";
    const int sms = get_num_sms();
    bool sized = false;
    for (;;) {
        NativeKernelArgs native{rendered_source<W4A8DispatchTag>(name, source, c),
            {dim3(c.num_sms), dim3(c.num_threads), 0},
            {native_pod_argument(symmetric_buffer(pointers, rank, c.num_ranks)), int32_t(tokens),
             w.data_ptr(), s.data_ptr(), inv, rw.data_ptr(), rs.data_ptr(), rinv,
             int32_t(w.size(0)), output.data_ptr(), limit}};
        const auto runtime = compile_native_kernel<W4A8DispatchTag>(name, native);
        native.launch_args.smem_size = native_kernel_smem(runtime);
        const int occupancy = native_kernel_occupancy(runtime, c.num_threads, native.launch_args.smem_size);
        TORCH_CHECK(occupancy >= 1, "W4A8 MegaMoE G1 cannot keep a block resident");
        const int resident_ctas = sms * std::min(8, occupancy);
        // The grid barrier's participant count is a template argument. Recompile
        // for the selected grid, then verify that specialization also fits.
        if (!sized || c.num_sms > resident_ctas) {
            sized = true;
            if (c.num_sms != resident_ctas) {
                c.num_sms = resident_ctas;
                continue;
            }
        }
        NativeKernelRuntime<W4A8DispatchTag>::launch(runtime, native);
        return;
    }
}

struct W4A8ScatterTag {};
inline void w4a8_scatter(const w4fa8::W4A8Config& tile, int groups,
    const torch::Tensor& a, const torch::Tensor& scales,
    const torch::Tensor& w, const torch::Tensor& s, float inv,
    const torch::Tensor& rw, const torch::Tensor& rs, float rinv,
    const torch::Tensor& rows, const torch::Tensor& valid, const torch::Tensor& routes,
    const torch::Tensor& combine, const torch::Tensor& offsets, int capacity,
    const std::optional<torch::Tensor>& scratch) {
    const int n = w.size(1), k = a.size(1), pool = a.size(0);
    auto* layout = rows.data_ptr<int32_t>();
    if (groups >= 128) {
        TORCH_CHECK(scratch && scratch->numel() >= 4 * (1 + groups + ceil_div(pool, tile.block_m)),
                    "W4A8 MegaMoE scheduler scratch is too small");
        auto args = ComputeBlockInfoKernelRuntime::Args{
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(layout), uint32_t(groups),
                                reinterpret_cast<uint32_t*>(scratch->data_ptr<int32_t>())},
            .launch_args = {1, std::max(32, next_power_of_two(groups)), 0}};
        const auto& runtime = compiler->build("computeBlockInfoKernel", ComputeBlockInfoKernelRuntime::generate(tile.block_m));
        ComputeBlockInfoKernelRuntime::launch(runtime, args);
        layout = scratch->data_ptr<int32_t>();
    }
    ScatterConfig c{};
    c.n = n; c.k = k; c.num_groups = groups;
    c.block_m = tile.block_m; c.block_n = tile.block_n; c.block_k = 128;
    c.warp_m = tile.warp_m; c.warp_n = 64; c.num_stages = tile.num_stages;
    c.num_threads = w4fa8::get_num_threads(tile);
    launch<W4A8ScatterTag>("w4a8_mega_moe_gemm2_scatter", R"(
#include <deep_gemm/impls/w4a8_mega_moe.cuh>
using namespace deep_gemm;
using Kernel = cutlass::gemm::kernel::W4A8GEMM_MMA<
    {N}, {K}, {BLOCK_M}, {BLOCK_N}, 128, {WARP_M}, 64, 128,
    {NUM_GROUPS}, {NUM_STAGES}, GemmType::MGroupedNoPad>;
extern "C" __device__ __constant__ uint32_t ppu_native_smem_bytes = sizeof(Kernel::SharedStorage);
extern "C" __global__ __launch_bounds__({NUM_THREADS}, 1)
void PPU_KERNEL_ENTRY(const cutlass::float_e4m3_t* a, const float* a_scale,
    const uint8_t* w, const uint8_t* s, float inv, const uint8_t* rw, const uint8_t* rs, float rinv,
    int masters, int32_t* rows, const int32_t* valid, const int64_t* routes,
    __ppu_bfloat16* output, const int64_t* offsets, int capacity, int pool) {
    extern __shared__ char smem[];
    Kernel::Params params{};
    params.ptr_a = a; params.ptr_b = w; params.ptr_scale = s;
    params.ptr_d = reinterpret_cast<cutlass::bfloat16_t*>(output);
    params.ptr_a_scale = a_scale;
    params.scheduler = TileSchedulerArguments(pool, rows);
    Kernel::TileScheduler scheduler(params.scheduler);
    const mega_moe::w4a8::Scatter<{N}, {K}, {BLOCK_M}, {BLOCK_N}> epilogue{
        output, a_scale, valid, routes, offsets, rw, rs, masters, capacity, inv, rinv};
    Kernel{}.run(params, smem, scheduler, epilogue);
}
)", c, {a.data_ptr(), scales.data_ptr(), w.data_ptr(), s.data_ptr(), inv,
        rw.data_ptr(), rs.data_ptr(), rinv, int32_t(w.size(0)), layout, valid.data_ptr(), routes.data_ptr(),
        combine.data_ptr(), offsets.data_ptr(), int32_t(capacity), int32_t(pool)},
        dim3(tile.num_sms), c.num_threads, true);
}

} // namespace deep_gemm::mega_moe_native
