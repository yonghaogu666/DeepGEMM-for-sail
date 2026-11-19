#pragma once
#include <deep_gemm/impls/fp8_gemm.cuh>
#include <deep_gemm/scheduler/sm89_mega_moe.cuh>
#include <deep_gemm/impls/moe_gemm_scatter.cuh>

namespace deep_gemm::mega_moe::fp8_blockwise {

// Each producer must finish its own remote stores before the later IPC
// barrier publishes readiness. Do not drain per tile: NCP writes may remain
// outstanding while this CTA computes its next GEMM tile.
template <class NativeKernel>
struct CompletedScatterKernel : NativeKernel {
    CUTLASS_DEVICE void operator()(const typename NativeKernel::Params& params, char* smem) {
        NativeKernel::operator()(params, smem);
#if __HGGC_ARCH__ == 150
        __ppu_wait_remote_vstcnt(0);
        __ppu_wait_vstcnt(0);
#endif
        __threadfence_system();
        __syncthreads();
    }
};

// Stage and scatter BF16 rows through shared memory. Rows covers
// the full tile or a reusable row segment, depending on the selected capacity.
template <class Native, class MainloopStorage, class Scatter, class Tile,
          int NumThreads, int Rows, int ShapeN, bool ReuseScatterSmem = false>
class BlockwiseScatterEpilogue {
    using Base = mega_moe::detail::MoeOverlapScatterEpilogue<
        Native, MainloopStorage, Scatter, Tile, NumThreads, ShapeN>;
public:
    using Arguments = typename Base::Arguments;
    using Params = typename Base::Params;
    using EpilogueSchedule = typename Base::EpilogueSchedule;
    using DispatchPolicy = typename Base::DispatchPolicy;
    using ThreadEpilogueOp = typename Base::ThreadEpilogueOp;
    using ElementOutput = typename Base::ElementOutput;
    using FragmentAccumulator = typename Base::FragmentAccumulator;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using ElementCompute = typename Base::ElementCompute;
    using ElementScalar = typename Base::ElementScalar;
    using ElementC = typename Base::ElementC;
    using StrideC = typename Base::StrideC;
    using ElementD = typename Base::ElementD;
    using StrideD = typename Base::StrideD;
    using GmemTiledCopyC = typename Base::GmemTiledCopyC;
    using GmemTiledCopyD = typename Base::GmemTiledCopyD;
    static constexpr int BlockN = cute::size<1>(Tile{});
    // DeepGemmUniversal overlays mainloop and epilogue storage.
    struct alignas(128) ScatterTileStorage {
        __ppu_bfloat16 scatter_tile[Rows * BlockN];
    };
    using SharedStorage = std::conditional_t<ReuseScatterSmem, ScatterTileStorage,
        mega_moe::detail::MoeOverlapBf16ScratchStorage<MainloopStorage, typename Native::SharedStorage, Rows * BlockN>>;
    using TensorStorage = SharedStorage;
    static constexpr int BlockM = cute::size<0>(Tile{});
    static_assert(Rows > 0 && BlockM % Rows == 0 && Rows % 16 == 0);
    static_assert((BlockN == 128 || BlockN == 256) && ShapeN % BlockN == 0);
    static_assert(sizeof(SharedStorage) <= 256 * 1024);

    template <class Problem>
    static constexpr Params to_underlying_arguments(
            Problem const& problem, Arguments const& args, void* workspace) {
        return Base::to_underlying_arguments(problem, args, workspace);
    }
    template <class Problem>
    static size_t get_workspace_size(Problem const&, Arguments const&) { return 0; }
    template <class Problem>
    static bool can_implement(Problem const&, Arguments const&) { return true; }

    CUTLASS_HOST_DEVICE BlockwiseScatterEpilogue(
            Params const& params, SharedStorage& storage)
        : params_(params), storage_(storage) {}
    CUTLASS_DEVICE bool is_source_needed() { return false; }

    template <class Problem, class BlockShape, class Coord, class Engine,
              class Layout, class Mma, class Residue>
    CUTLASS_DEVICE void operator()(
            Problem, BlockShape, Coord block_coord,
            cute::Tensor<Engine, Layout> const& accum, Mma mma,
            Residue residue, int thread, char*) {
        using namespace cute;
        const unsigned row_start = unsigned(get<0>(block_coord)) * BlockM;
        const unsigned n_start = unsigned(get<1>(block_coord)) * BlockN;
        const unsigned pool_base =
            unsigned((params_.ptr_D - params_.base_ptr_D) / ShapeN) + row_start;
        // Blockwise GEMM2 schedules exact expert rows. Pool padding only changes the address.
        const unsigned valid = min(unsigned(BlockM), unsigned(get<0>(residue)));
        auto coords = mma.get_thread_slice(thread).partition_C(
            make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{})));
        static_assert(decltype(size<0>(shape<0>(coords)))::value == 2);
        using ExpectedLayout = cute::Layout<
            Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
            Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
        static_assert(cute::is_same_v<typename Mma::AtomLayoutC_TV, ExpectedLayout>);
        // The innermost fragment is an adjacent N pair, as in the existing
        // full-tile scatter epilogue; preserve the same BF16 rounding.
        constexpr auto probe_coords = Mma{}.get_thread_slice(_0{}).partition_C(
            make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockN>{})));
        constexpr auto probe_fragment = Mma::make_fragment_C(probe_coords);
        CUTE_STATIC_ASSERT_V((stride<0, 0>(probe_fragment) == _1{}));
        using Pair = cutlass::Array<float, 2>;
        union Packed { __ppu_bfloat162 bf16; uint32_t bits; };

        // Full-tile scratch uses one compile-time iteration. Keep the valid-row
        // loop rolled for smaller scratch to avoid BM192 register spills.
        if constexpr (ReuseScatterSmem) {
            // Retire mainloop reads and AIU copies before overwriting the same storage.
            cute::cp_async_wait<0>();
            __syncthreads();
        }
        #pragma unroll 1
        for (unsigned chunk = 0;
             chunk < (Rows == BlockM ? unsigned(BlockM) : valid); chunk += Rows) {
            cute::for_each(cute::make_seq<CUTE_STATIC_V(size<1>(coords))>{}, [&](auto rm) {
                cute::for_each(cute::make_seq<CUTE_STATIC_V(size<1>(shape<0>(coords)))>{}, [&](auto am) {
                    auto row_coord = coords(make_coord(_0{}, am, _0{}), rm, _0{});
                    unsigned row = unsigned(get<0>(row_coord));
                    if (row >= chunk && row < chunk + Rows && row < valid) {
                        cute::for_each(cute::make_seq<CUTE_STATIC_V(size<2>(coords))>{}, [&](auto rn) {
                            cute::for_each(cute::make_seq<CUTE_STATIC_V(size<2>(shape<0>(coords)))>{}, [&](auto an) {
                                auto coord = coords(make_coord(_0{}, am, an), rm, rn);
                                const Pair values = *reinterpret_cast<const Pair*>(
                                    accum(make_coord(_, am, an), rm, rn).data());
                                Packed packed;
                                packed.bf16 = __floats2bfloat162_rn(values[0], values[1]);
                                cutlass::arch::shared_store<4>(
                                    cutlass::arch::cutlass_get_smem_pointer(
                                        storage_.scatter_tile + (row - chunk) * BlockN
                                        + unsigned(get<1>(coord))), &packed.bits);
                            });
                        });
                    }
                });
            });
#if __HGGC_ARCH__ == 150
            if constexpr (Rows == BlockM)
                __ppu_barrier_sync_nocnt(0, 1);
            else
                __syncthreads();
#else
            __syncthreads();
#endif
            Scatter::template scatter_staged_bf16<Rows, BlockN, NumThreads, BlockN == 128>(
                params_.scatter, storage_.scatter_tile,
                valid > chunk ? min(unsigned(Rows), valid - chunk) : 0,
                n_start, pool_base + chunk);
            // Finish shared loads before reusing scratch. Remote writes can
            // remain posted until CompletedScatterKernel drains the CTA.
            if constexpr (Rows < BlockM || ReuseScatterSmem)
                __syncthreads();
        }
    }
private:
    Params params_;
    SharedStorage& storage_;
};

// Schedule actual expert rows, independently of the 128-row landing-pool
// padding. Reuse native tile ordering, including its large-group metadata
// preprocessing, and restore the physical pool offset for each selected tile.
template <int N, int K, int BlockM, int BlockN, int Groups, int Ranks, bool WarpSchedule = false>
struct ValidRowsPoolScheduler
    : sched::RoutedMegaMoEScheduler<N, K, BlockM, BlockN, Groups, Ranks> {
    using Base = sched::RoutedMegaMoEScheduler<N, K, BlockM, BlockN, Groups, Ranks>;
    struct Params : Base::Params {
        const int32_t* pool_row_offsets{nullptr};
        CUTLASS_HOST_DEVICE Params() : Base::Params() {}
        CUTLASS_HOST_DEVICE Params(const typename Base::Params& other) : Base::Params(other) {}
    };
    const int32_t* pool_row_offsets;
    CUTLASS_DEVICE explicit ValidRowsPoolScheduler(const Params& params, int warp_group = 0)
        : Base(params, warp_group), pool_row_offsets(params.pool_row_offsets) {}
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m, uint32_t& n) {
        if (!Base::fetch_next_work(m, n)) return false;
        this->curr_cumsum_m = __ldg(pool_row_offsets + this->curr_group_idx);
        return true;
    }
};

template <int N, int K, int BlockM, int BlockN, int Groups, int Ranks>
struct ValidRowsPoolScheduler<N, K, BlockM, BlockN, Groups, Ranks, true>
    : ValidRowsPoolScheduler<N, K, BlockM, BlockN, Groups, Ranks> {
    using Base = ValidRowsPoolScheduler<N, K, BlockM, BlockN, Groups, Ranks>;
    using Params = typename Base::Params;
    static_assert(Groups <= 64);
    sched::DistributedMegaMoEBlocks<Groups, BlockM> blocks;
    static constexpr int kChunks = (Groups + 31) / 32;
    uint32_t lane_offsets[kChunks];
    CUTLASS_DEVICE explicit ValidRowsPoolScheduler(const Params& params, int warp_group = 0)
        : Base(params, warp_group), blocks(params.grouped_layout, threadIdx.x % 32) {
        const uint32_t lane = threadIdx.x % 32;
        #pragma unroll
        for (int chunk = 0; chunk < kChunks; ++chunk) {
            const uint32_t expert = chunk * 32 + lane;
            lane_offsets[chunk] = expert < Groups ? uint32_t(__ldg(this->pool_row_offsets + expert)) : 0;
        }
    }
    CUTLASS_DEVICE bool fetch_next_work(uint32_t& m, uint32_t& n) {
        const uint32_t work = this->current_iter++ * gridDim.x + blockIdx.x;
        uint4 entry;
        if (!blocks.get_block(work / Base::kNumNBlocks, entry)) return false;
        this->curr_group_idx = entry.x;
        this->curr_group_m = entry.y;
        uint32_t offset = lane_offsets[0];
        if constexpr (kChunks == 2)
            if (entry.x >= 32) offset = lane_offsets[1];
        this->curr_cumsum_m = __shfl_sync(0xffffffffu, offset, entry.x % 32);
        this->get_swizzled_block_idx(math::ceil_div(entry.y, uint32_t(BlockM)),
                                     work - entry.z * Base::kNumNBlocks, m, n);
        return true;
    }
};

template <int N, int K, int Groups, int MasterGroups, int Stages, int BlockM, int ScratchRows, int Ranks, int BlockN,
          bool WarpSchedule = false, bool ReuseScatterSmem = false>
struct Gemm2Scatter {
    // Native blockwise mainloop and GroupedNoPad scheduler, with the existing
    // shared MoE epilogue adapter. The checkpoint scales remain 128x128.
    static_assert((BlockM == 32 || BlockM == 64 || BlockM == 128 || BlockM == 192) &&
                  Stages >= 2 && Stages <= 4);
    static constexpr int WarpM = BlockM == 64 ? 32 : (BlockM < 64 ? 16 : BlockM / 4);
    static constexpr int WarpN = BlockN / 4;
    using Tile = cute::Shape<cute::Int<BlockM>, cute::Int<BlockN>, cute::_128>;
    using Warp = cute::Shape<cute::Int<WarpM>, cute::Int<WarpN>, cute::_128>;
    using ScaleShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig = decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<
        ScaleShape, false, true>(ScaleShape{}));
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::PPU0015, cutlass::arch::OpClassTensorOp,
        cutlass::float_e4m3_t, cute::tuple<cutlass::layout::RowMajor, decltype(ScaleConfig::deduce_layoutSFA())>, 1,
        cutlass::float_e4m3_t, cute::tuple<cutlass::layout::ColumnMajor, decltype(ScaleConfig::deduce_layoutSFB())>, 1,
        float, Tile, Warp, cute::Int<Stages>,
        cutlass::gemm::KernelAiuMultistageWithBlockWiseScale>::CollectiveOp;
    using Stride = cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>;
    using NativeEpilogue = cutlass::epilogue::collective::DefaultEpilogueNoTsm<
        Stride, Stride,
        cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 2, float, float,
            cutlass::epilogue::thread::ScaleType::Nothing>,
        cutlass::gemm::EpilogueDefault, N % BlockN == 0>;
    static constexpr int kNumThreads = cute::size(typename Mainloop::TiledMma{});
    using Epilogue = BlockwiseScatterEpilogue<
        NativeEpilogue, typename Mainloop::SharedStorage,
        mega_moe::detail::Bf16MoeScatter<>, Tile, kNumThreads,
        (ScratchRows > 0 ? ScratchRows : BlockM), N, ReuseScatterSmem>;
    using Scheduler = ValidRowsPoolScheduler<N, K, BlockM, BlockN, Groups, Ranks, WarpSchedule>;
    using Kernel = CompletedScatterKernel<DeepGemmUniversal<
        Shape<int, int, int, int>, Mainloop, Epilogue, Scheduler, false, false, true>>;

};

}  // namespace deep_gemm::mega_moe::fp8_blockwise
