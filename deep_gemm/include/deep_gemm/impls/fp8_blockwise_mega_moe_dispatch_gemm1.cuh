#pragma once
#include <deep_gemm/impls/mega_moe_io.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/epilogue/gated_activation.cuh>
#include <deep_gemm/epilogue/activation_quant_ops.cuh>
#include <tools/util/include/cutlass/util/packed_stride.hpp>
#include <deep_gemm/scheduler/mega_moe_work_queue.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>

// Native block-FP8 fused dispatch/GEMM1.
#include <deep_gemm/impls/mega_moe_transport.cuh>
#include <deep_gemm/impls/fp8_gemm.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

constexpr int kActivationRowStride = 258;

// Gate/up retain native 128-row blockwise weight-scale groups.
template <int kN, int kK, int kComputeM, class Builder>
struct Gemm1 {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    static constexpr int kTileN = Builder::BLOCK_N;
    static_assert(kTileN == 128 || kTileN == 256);
    using TileShape = cute::Shape<cute::Int<kComputeM>, cute::Int<kTileN>, cute::_128>;
    static constexpr int kWarpM = kComputeM == 64 ? 32 : kComputeM < 64 ? 16 : kComputeM / 4;
    static constexpr int kWarpN = 64;
    using WarpShape = cute::Shape<cute::Int<kWarpM>, cute::Int<kWarpN>, cute::_128>;
    using ScaleShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<
        ScaleShape, false, true>(ScaleShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::PPU0015, cutlass::arch::OpClassTensorOp,
        ElementA, cute::tuple<cutlass::layout::RowMajor, LayoutSFA>, 1,
        ElementB, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>, 1,
        float, TileShape, WarpShape, cute::Int<Builder::STAGES>,
        cutlass::gemm::KernelAiuMultistageWithBlockWiseScale>::CollectiveOp;
    static constexpr int kNumThreads = cute::size(typename CollectiveMainloop::TiledMma{});

    template <int kExperts>
    static constexpr uint32_t shared_memory_bytes() {
        // Expert counts, pull staging and GEMM execute in separate phases and
        // reuse the same allocation. GEMM1's epilogue also fits its mainloop storage.
        constexpr uint32_t gemm = sizeof(typename CollectiveMainloop::SharedStorage);
        constexpr uint32_t pull = (kNumThreads / 32) * kK;
        constexpr uint32_t counts = kExperts * sizeof(uint32_t);
        constexpr uint32_t data = gemm > pull ? gemm : pull;
        return data > counts ? data : counts;
    }
};

template <class Accum, class Coordinates>
__device__ __forceinline__ void store_g1_panel(Accum& accum, const Coordinates& coords,
        float* tile, int valid_rows, int col_base) {
    using namespace cute;
    constexpr int kStride = kActivationRowStride;
    CUTLASS_PRAGMA_UNROLL
    for (int wm = 0; wm < size<1>(coords); ++wm) {
        CUTLASS_PRAGMA_UNROLL
        for (int mm = 0; mm < size<1>(shape<0>(coords)); ++mm) {
            const int base_m = get<0>(coords(make_coord(Int<0>{}, mm, Int<0>{}), wm, Int<0>{}));
            CUTLASS_PRAGMA_UNROLL
            for (int wn = 0; wn < size<2>(coords); ++wn) {
                CUTLASS_PRAGMA_UNROLL
                for (int mn = 0; mn < size<2>(shape<0>(coords)); ++mn) {
                    auto frag = accum(make_coord(_, mm, mn), wm, wn);
                    const int col = get<1>(coords(make_coord(Int<0>{}, mm, mn), wm, wn)) - col_base;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < size(frag) / 2; ++i) {
                        const int row = base_m + i * 8;
                        if (row < valid_rows)
                            *reinterpret_cast<float2*>(tile + row * kStride + col) =
                                make_float2(frag(2 * i), frag(2 * i + 1));
                    }
                }
            }
        }
    }
}

template <int kN, int kNumThreads, bool kClamp, GatedActivation Activation,
          bool kWritePaddedRows, int lanes, class PrefetchTicket>
__device__ __forceinline__ void quantize_g1_shared(float* tile, uint8_t* output, float* scales,
        int pool_rows, int valid_rows, int n_block, int thread, float swiglu_limit,
        float activation_alpha, float situ_beta, float situ_linear_beta,
        int writable_rows, const PrefetchTicket& prefetch_ticket) {
    using namespace cute;
    constexpr int kStride = kActivationRowStride;
    __syncthreads();
    prefetch_ticket();
    static_assert(lanes == 8 || lanes == 16 || lanes == 32);
    constexpr int elements = 128 / lanes;
    using PackedOutput = std::conditional_t<lanes == 8, uint4, std::conditional_t<lanes == 16, uint2, uint32_t>>;
    const int lane = thread % lanes;
    const int rows_to_write = kWritePaddedRows ? writable_rows : valid_rows;
    for (int row = thread / lanes; row < rows_to_write; row += kNumThreads / lanes) {
        float values[elements];
        float amax = 1e-10f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < elements / 2; ++i) {
            float2 activated = make_float2(0.0f, 0.0f);
            if (!kWritePaddedRows || row < valid_rows) {
                float2 g = *reinterpret_cast<const float2*>(
                    tile + row * kStride + lane * elements + i * 2);
                float2 u = *reinterpret_cast<const float2*>(
                    tile + row * kStride + 128 + lane * elements + i * 2);
                activated = make_float2(
                    routed_gated_activation<Activation, bool(DG_MOE_FAST_MATH)>(
                        g.x, u.x, kClamp ? swiglu_limit : -1.0f, activation_alpha, situ_beta, situ_linear_beta),
                    routed_gated_activation<Activation, bool(DG_MOE_FAST_MATH)>(
                        g.y, u.y, kClamp ? swiglu_limit : -1.0f, activation_alpha, situ_beta, situ_linear_beta));
            }
            values[2 * i] = activated.x;
            values[2 * i + 1] = activated.y;
            amax = fmaxf(amax, fmaxf(fabsf(activated.x), fabsf(activated.y)));
        }
        amax = epilogue::quant::warp_amax<lanes>(amax);
        const auto quant_scale = epilogue::quant::fp8_block_scale<bool(DG_MOE_FAST_MATH)>(amax);
        if (lane == 0)
            scales[int64_t(n_block) * pool_rows + row] = quant_scale.scale;
        __align__(16) uint16_t packed[elements / 2];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < elements / 2; ++i)
            packed[i] = epilogue::quant::fp8_pair(values[2 * i] * quant_scale.inverse,
                                                values[2 * i + 1] * quant_scale.inverse);
        *reinterpret_cast<PackedOutput*>(output + int64_t(row) * (kN / 2) + n_block * 128 + lane * elements) =
            *reinterpret_cast<PackedOutput*>(packed);
    }
    __syncthreads();
}

template <int kN, int kK, bool kClamp, GatedActivation Activation, int kComputeM,
          class Builder, class PrefetchTicket>
__device__ __forceinline__ void gemm1_block(
        const uint8_t* a, const float* sa, const uint8_t* b, const float* sb,
        uint8_t* output, float* output_scales, int pool_rows, int valid_rows, int n_block,
        int thread, char* smem, float swiglu_limit, float activation_alpha, float situ_beta,
        float situ_linear_beta, int writable_rows, const PrefetchTicket& prefetch_ticket) {
    using namespace cute;
    using G = Gemm1<kN, kK, kComputeM, Builder>;
    using Mainloop = typename G::CollectiveMainloop;
    using Tile = typename G::TileShape;
    using Mma = typename Mainloop::TiledMma;
    using StrideA = typename Mainloop::StrideA;
    using StrideB = typename Mainloop::StrideB;
    auto problem = make_shape(valid_rows, kN, kK, 1);
    typename Mainloop::Params params{
        reinterpret_cast<const typename G::ElementA*>(a),
        cutlass::make_cute_packed_stride(StrideA{}, make_shape(valid_rows, kK, 1)),
        reinterpret_cast<const typename G::ElementB*>(b),
        cutlass::make_cute_packed_stride(StrideB{}, make_shape(kN, kK, 1)), 4,
        sa, G::ScaleConfig::tile_atom_to_shape_SFA(make_shape(pool_rows, kN, kK, 1)),
        sb, G::ScaleConfig::tile_atom_to_shape_SFB(make_shape(pool_rows, kN, kK, 1))};
    Mainloop mainloop;
    Mma mma;
    auto gate = partition_fragment_C(mma, take<0, 2>(Tile{}));
    const auto project = [&](auto& accum, int panel) {
        auto inputs = mainloop.load_init(problem, make_coord(0, panel, _, 0), params);
        auto gA = get<0>(inputs);
        clear(accum);
        auto k_iter = make_coord_iterator(shape<2>(gA));
        mainloop(params, inputs, accum, k_iter, int(size<2>(gA)), thread, smem);
        __syncthreads();
    };
    static_assert(kComputeM * kActivationRowStride * sizeof(float) <= sizeof(typename Mainloop::SharedStorage));
    auto* shared = reinterpret_cast<float*>(smem);
    constexpr int panels = 256 / G::kTileN;
    const int first_panel = n_block * panels;
    const auto store = [&](auto& accum, int panel, int col_offset) {
        auto identity = make_identity_tensor(make_shape(valid_rows, kN));
        auto tile = local_tile(identity, take<0, 2>(Tile{}), make_coord(0, panel));
        store_g1_panel(accum, mma.get_thread_slice(thread).partition_C(tile),
                       shared + col_offset, valid_rows, panel * G::kTileN);
    };
    project(gate, first_panel);
    if constexpr (panels == 2) {
        // Keep gate in registers while the up projection still owns shared memory.
        auto up = partition_fragment_C(mma, take<0, 2>(Tile{}));
        project(up, first_panel + 1);
        store(gate, first_panel, 0);
        store(up, first_panel + 1, G::kTileN);
    } else {
        store(gate, first_panel, 0);
    }
    if constexpr (kClamp)
        swiglu_limit = __bfloat162float(__float2bfloat16(swiglu_limit));
    quantize_g1_shared<kN, G::kNumThreads, kClamp, Activation, Builder::WRITE_PADDED_ROWS, Builder::QUANT_LANES>(
        shared, output, output_scales, pool_rows, valid_rows, n_block, thread,
        swiglu_limit, activation_alpha, situ_beta, situ_linear_beta, writable_rows, prefetch_ticket);
}

template <class Builder, uint32_t kRanks, uint32_t kCTAs, uint32_t kExperts,
          uint32_t kTopk, uint32_t kHidden, uint32_t kN, uint32_t kPullCTAs,
          bool kClamp = false,
          uint32_t kMasterExperts = kExperts / kRanks, GatedActivation Activation = GatedActivation::SwiGLU,
          uint32_t kComputeM = 128, uint32_t kGemm2BlockM = 0>
CUTLASS_DEVICE void dispatch_gemm1_kernel_device(
        void* workspace_base, const layout::SymBuffer<>& sym_buffer,
        int64_t* ids, const uint8_t* input,
        const float* scales, uint8_t* pool, float* pool_scales,
        int32_t* gemm2_metadata, uint4* gemm2_block_info, const uint8_t* w1, const float* s1,
        uint8_t* output, float* output_scales,
        uint32_t num_tokens, uint32_t cap, float swiglu_limit = 0.0f, float activation_alpha = 1.0f, float situ_beta = 4.0f, float situ_linear_beta = 25.0f,
        const uint8_t* w1_replicas = nullptr, const float* s1_replicas = nullptr,
        uint64_t routing_bytes = 0) {
    static_assert(kRanks > 0 && kRanks <= 64 && kExperts % kRanks == 0);
    constexpr uint32_t kEpr = kExperts / kRanks;
    constexpr uint32_t kBlockM = 128, kBlockN = 256;
    using ScheduledGemm = Gemm1<kN, kHidden, kComputeM, Builder>;
    constexpr uint32_t kWorkBatch = Builder::WORK_BATCH;
    constexpr uint32_t kThreads = ScheduledGemm::kNumThreads;
    constexpr uint32_t kWarps = kThreads / 32, kScales = kHidden / 128;
    constexpr uint32_t kNBlocks = kN / kBlockN;
    constexpr bool kCompactDecode = Builder::PULL == mega::PullSchedule::Compact;
    constexpr bool kPipelinedPull = Builder::PULL != mega::PullSchedule::Direct;
    static_assert(!kPipelinedPull || 2 * kWarps * kHidden <=
        ScheduledGemm::template shared_memory_bytes<kExperts>(),
        "Two row slots per warp must fit inside the original dynamic shared allocation");
    // The interleave vote and shuffle cover one warp of experts.
    constexpr bool kAsyncScales = kPipelinedPull && (kCompactDecode || kComputeM == 64);
    constexpr int kExpertStripe = 4;
    constexpr bool kInterleaveExperts = kAsyncScales && kEpr < 32 && kNBlocks % kExpertStripe == 0;
    constexpr uint32_t kPullPayloadBytes = 2 * kWarps * kHidden;
    // Keep each scale row aligned for 16-byte copies.
    constexpr uint32_t kScaleScratchStride = kScales + 4;
    static_assert(!kAsyncScales || (kScales % 4 == 0 && kPullPayloadBytes % 16 == 0 &&
        kPullPayloadBytes + kBlockM * kScaleScratchStride * sizeof(float) <=
            ScheduledGemm::template shared_memory_bytes<kExperts>()));
    static_assert(kHidden % 128 == 0 && kN % kBlockN == 0);
    static_assert(kEpr > 0 && kEpr <= 128);
    static_assert(kMasterExperts > 0 && kMasterExperts <= kEpr);
    static_assert(kPullCTAs > 0 && kPullCTAs < kCTAs);
    extern __shared__ __align__(128) unsigned char smem[];
    const uint32_t sm = blockIdx.x, thread = threadIdx.x;
    const uint32_t warp = thread / 32, lane = thread % 32;
    const layout::Workspace ws(workspace_base, kRanks, kExperts, cap, kTopk, routing_bytes);
    static_assert(kWorkBatch > 0);
    if (sm == 0 && thread == 0)
        *ws.get_gemm1_work_count_ptr() = (kCTAs - kPullCTAs) * kWorkBatch;
    for (uint32_t g = sm * kThreads + thread;
         g < ws.num_max_pool_tokens / kBlockM; g += kCTAs * kThreads)
        *ws.get_l1_arrival_count_ptr(g) = 0;
    mega_moe::detail::dispatch_handshake<kRanks, kCTAs, kThreads, kExperts, kEpr, kTopk, true>(
        ws, sym_buffer, ids, reinterpret_cast<uint32_t*>(smem), num_tokens, sm, thread);

    const sched::DistributedMegaMoEBlocks<kEpr, kBlockM> blocks(ws.get_expert_recv_count_sum_ptr(), lane);
    const uint32_t total_blocks = blocks.total_blocks;
    // With one landing block and one compute block per live expert, stripe
    // the first work across all experts instead of concentrating the waits.
    bool striped = false;
    if constexpr (kInterleaveExperts) {
        const auto rows = blocks.rows[0];
        striped = __all_sync(0xffffffffu, lane >= kEpr || (rows <= kComputeM && (kCompactDecode || rows > 0)));
    }
    constexpr bool compact_experts = Builder::WORK == mega::Gemm1WorkSchedule::DynamicSingleTile;
    bool single_tile = true;
    if constexpr (compact_experts) {
        CUTLASS_PRAGMA_UNROLL
        for (uint32_t i = 0; i < decltype(blocks)::kChunks; ++i)
            single_tile &= __all_sync(0xffffffffu, blocks.rows[i] <= kComputeM);
    }
    const layout::Gemm2Metadata gemm2_meta(gemm2_metadata, kEpr);
    if (sm == 0 && warp == 0)
        gemm2_meta.store_expert_layout<kEpr, kBlockM, kGemm2BlockM>(blocks, lane, gemm2_block_info);
    static_assert(kComputeM == 32 || kComputeM == 64 || kComputeM == 128 || kComputeM == 192);
    auto* work_counter = ws.get_gemm1_work_count_ptr();
    constexpr bool static_work = Builder::WORK == mega::Gemm1WorkSchedule::Static;
    constexpr bool kPrefetchTicket = !static_work && kAsyncScales;
    constexpr uint32_t kTicketOffset = kComputeM * kActivationRowStride * sizeof(float);
    static_assert(!kPrefetchTicket || (kWorkBatch == 1 &&
        kTicketOffset % alignof(uint32_t) == 0 && kTicketOffset + sizeof(uint32_t) <=
        ScheduledGemm::template shared_memory_bytes<kExperts>()));
    // Keep the ticket outside both BM32 and BM64 activation tiles. The
    // epilogue publishes it and the input-ready barrier retires its readers.
    const mega_moe::WorkQueue queue{work_counter,
        reinterpret_cast<uint32_t*>(smem + (kPrefetchTicket ? kTicketOffset : 0))};
    const auto prefetch_ticket = [&] {
        if (kPrefetchTicket && thread == kThreads - 32)
            queue.claim(0);
    };

    struct TileInfo { uint32_t expert, begin, valid, local_begin; };
    const auto resolve = [&](uint32_t g) {
        uint4 info;
        blocks.get_block(g, info);
        const uint32_t expert = info.x, rows = info.y, begin = info.z;
        const uint32_t local_begin = (g - begin) * kBlockM;
        return TileInfo{expert, g * kBlockM, mega_moe::detail::umin(kBlockM, rows - local_begin), local_begin};
    };

    if (sm < kPullCTAs) {
        auto* scratch = smem + warp * kHidden * (kPipelinedPull ? 2 : 1);
        auto* scale_scratch = reinterpret_cast<float*>(smem + kPullPayloadBytes);
        // Each producer owns complete M tiles. Different producers can pull
        // subsequent tiles while consumer CTAs compute already-published ones.
        for (uint32_t g = sm; g < total_blocks; g += kPullCTAs) {
            const auto tile = resolve(g);
            const mega_moe::detail::WarpSourceRanges<kRanks> sources(ws, tile.expert, lane);
            const auto stage_row = [&](uint32_t row, uint8_t* staging) {
                const uint32_t in_expert = tile.local_begin + row;
                const auto source = sources.resolve(in_expert, lane);
                const uint32_t src = source.rank, slot = source.slot;
                uint32_t route = lane == 0 ? *ws.get_src_token_topk_idx_ptr(tile.expert, src, slot) : 0;
                route = __shfl_sync(0xffffffffu, route, 0);
                const uint32_t token = route / kTopk, topk_slot = route % kTopk;
                const uint32_t dst = tile.begin + row;
                const auto* src_scales = sym_buffer.map(scales + uint64_t(token) * kScales, src);
                if constexpr (kAsyncScales) {
                    const auto* src_vec = reinterpret_cast<const uint4*>(src_scales);
                    auto* dst_vec = reinterpret_cast<uint4*>(scale_scratch + row * kScaleScratchStride);
                    for (uint32_t vec = lane; vec < kScales / 4; vec += 32)
                        cute::PPU_CP_ASYNC_CACHEALWAYS<uint4>::copy(src_vec[vec], dst_vec[vec]);
                }
                // This closes the async group for both payload and scale copies.
                mega_moe::detail::warp_async_copy_global_to_shared(staging,
                    sym_buffer.map(input + uint64_t(token) * kHidden, src), kHidden, lane);
                if constexpr (!kAsyncScales) {
                    for (uint32_t sf = lane; sf < kScales; sf += 32)
                        pool_scales[uint64_t(sf) * ws.num_max_pool_tokens + dst] = src_scales[sf];
                }
                if (lane == 0)
                    gemm2_meta.store_route(dst, src, token, topk_slot);
            };
            const auto store_row = [&](uint32_t row, const uint8_t* slot) {
                __syncwarp();
                const uint32_t dst = tile.begin + row;
                mega_moe::detail::warp_copy_bytes(
                    pool + uint64_t(dst) * kHidden, slot, kHidden, lane);
                __syncwarp();
            };
            if constexpr (kPipelinedPull) {
                if (warp < tile.valid) {
                    uint32_t row = warp;
                    uint8_t* current = scratch;
                    uint8_t* next = scratch + kHidden;
                    stage_row(row, current);
                    for (; row + kWarps < tile.valid; row += kWarps) {
                        stage_row(row + kWarps, next);
                        // The newer group may remain in flight. Do not recycle
                        // the previous slot until every lane retires its read.
                        cute::cp_async_wait<1>();
                        store_row(row, current);
                        auto* retired = current;
                        current = next;
                        next = retired;
                    }
                    cute::cp_async_wait<0>();
                    store_row(row, current);
                }
            } else {
                for (uint32_t row = warp; row < tile.valid; row += kWarps) {
                    stage_row(row, scratch);
                    cute::cp_async_wait<0>();
                    store_row(row, scratch);
                }
            }
            if constexpr (kAsyncScales) {
                // Every warp has retired its final copy group. Transpose to
                // the native group-major pool with contiguous row stores.
                __syncthreads();
                for (uint32_t sf = warp; sf < kScales; sf += kWarps)
                    for (uint32_t row = lane; row < tile.valid; row += 32)
                        pool_scales[uint64_t(sf) * ws.num_max_pool_tokens + tile.begin + row] =
                            scale_scratch[row * kScaleScratchStride + sf];
            }
#if __HGGC_ARCH__ == 150
            __ppu_wait_vstcnt(0);
#endif
            __threadfence();
            __syncthreads();
            if (thread == 0)
                tix::red_add_rel(ws.get_l1_arrival_count_ptr(g), 1u);
        }
    }
    {
        mega_moe::ArrivalCache ready;
        uint32_t work = (sm - kPullCTAs) * kWorkBatch;
        if constexpr (static_work) {
            work = (sm + kCTAs - kPullCTAs) % kCTAs;
        } else if (sm < kPullCTAs) {
            queue.prime(thread, kWorkBatch);
            work = queue.current();
        }
        uint32_t batch_end = work + kWorkBatch;
        sched::MegaMoEGemm1Scheduler scheduler;
        sched::RowCountMegaMoEBlocks<kEpr, kComputeM, kBlockM, uint64_t> metadata{
            ws.get_expert_recv_count_sum_ptr()};
        uint32_t m, n;
        const auto select_work = [&] {
            if constexpr (compact_experts) {
                // With one compute tile per live expert the landing-block index is exact.
                if (single_tile)
                    return scheduler.select_work<kComputeM, kNBlocks>(work, m, n, blocks);
            }
            if constexpr (kInterleaveExperts) {
                if (striped) {
                    // When every active expert fits one M tile, its
                    // compact index is also its physical pool-block index.
                    const uint32_t experts = kCompactDecode ? total_blocks : kEpr;
                    if (work >= experts * kNBlocks)
                        return false;
                    const uint32_t stripe_work = work / kExpertStripe;
                    uint32_t stripe_index = stripe_work / kEpr;
                    uint32_t active_index = stripe_work % kEpr;
                    uint32_t expert = active_index;
                    if constexpr (kCompactDecode) {
                        if (experts != kEpr) {
                            stripe_index = stripe_work / experts;
                            active_index = stripe_work - stripe_index * experts;
                            const auto matches = __ballot_sync(0xffffffffu,
                                lane < kEpr && blocks.rows[0] > 0 && blocks.begin[0] == active_index);
                            expert = uint32_t(__ffs(matches) - 1);
                        }
                    }
                    n = stripe_index * kExpertStripe + work % kExpertStripe;
                    m = 0;
                    scheduler.curr_group_idx = expert;
                    scheduler.curr_group_m = __shfl_sync(0xffffffffu, blocks.rows[0], expert);
                    scheduler.curr_cumsum_m = active_index * kBlockM;
                    return true;
                }
            }
            return scheduler.select_work<kComputeM, kNBlocks>(work, m, n, metadata);
        };
        #pragma clang loop licm(disable)
        while (select_work()) {
            const uint32_t expert = scheduler.curr_group_idx, rows = scheduler.curr_group_m;
            const uint32_t dispatch_blocks = math::ceil_div(rows, kBlockM);
            const uint32_t local_begin = m * kComputeM;
            const uint32_t valid = mega_moe::detail::umin(kComputeM, rows - local_begin);
            const uint32_t base = scheduler.curr_cumsum_m + local_begin;
            const uint32_t first = base / kBlockM, last = (base + valid - 1) / kBlockM;
            ready.wait(ws.get_l1_arrival_count_ptr(0), first, last, total_blocks);
            asm volatile("" ::: "memory");
            uint32_t weight_expert = expert;
            const uint8_t* weight_base = w1;
            const float* scale_base = s1;
            if constexpr (kMasterExperts < kEpr) {
                if (weight_expert >= kMasterExperts) {
                    weight_expert -= kMasterExperts;
                    weight_base = w1_replicas;
                    scale_base = s1_replicas;
                }
            }
            const uint32_t writable = local_begin + kComputeM >= rows
                ? dispatch_blocks * kBlockM - local_begin : kComputeM;
            auto* tile_output = output + uint64_t(base) * (kN / 2);
            const auto run_compute = [&](auto compute_m) {
                gemm1_block<kN, kHidden, kClamp, Activation, decltype(compute_m)::value, Builder>(
                    pool + uint64_t(base) * kHidden, pool_scales + base,
                    weight_base + uint64_t(weight_expert) * kN * kHidden,
                    scale_base + uint64_t(weight_expert) * (kN / 128) * kScales,
                    tile_output, output_scales + base,
                    ws.num_max_pool_tokens, valid, n, thread, reinterpret_cast<char*>(smem),
                    swiglu_limit, activation_alpha, situ_beta, situ_linear_beta, writable, prefetch_ticket);
            };
            if constexpr (kComputeM == 64 && !Builder::WRITE_PADDED_ROWS) {
                using SmallGemm = Gemm1<kN, kHidden, 32, Builder>;
                static_assert(kThreads == SmallGemm::kNumThreads);
                static_assert(SmallGemm::template shared_memory_bytes<kExperts>() <=
                              ScheduledGemm::template shared_memory_bytes<kExperts>());
                // Keep one weight load when an expert only slightly exceeds BM32.
                if (valid <= 32)
                    run_compute(cute::Int<32>{});
                else
                    run_compute(cute::Int<64>{});
            } else {
                run_compute(cute::Int<kComputeM>{});
            }
            if constexpr (static_work) {
                work += kCTAs;
            } else if (++work == batch_end) {
                if (!kPrefetchTicket)
                    queue.prime(thread, kWorkBatch);
                work = queue.current();
                batch_end = work + kWorkBatch;
            }
        }
    }
}

} // namespace deep_gemm::mega_moe::fp8_blockwise
