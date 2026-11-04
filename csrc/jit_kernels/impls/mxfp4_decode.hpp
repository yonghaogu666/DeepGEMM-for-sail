#pragma once
#include <torch/python.h>
#include <c10/core/DeviceGuard.h>
#include <cstdint>
#include <limits>
#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"

namespace deep_gemm::w4fa8 {

class Mxfp4DecodeRuntime final : public LaunchRuntime<Mxfp4DecodeRuntime> {
public:
    static constexpr int kThreads = 128;
    static constexpr int kItemsPerThread = 8;

    struct Args {
        const void* packed;
        const void* indices;
        void* output;
        uint32_t count;
        LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args&) {
        return fmt::format(R"(
#include <deep_gemm/impls/mxfp4_decode.cuh>
extern "C" __global__ __launch_bounds__({})
void dequantize_mxfp4_fp8(const uint4* packed, const uint8_t* indices, uint4* output, uint32_t count) {{
    deep_gemm::w4fa8::decode_mxfp4_fp8<{}, {}>(packed, indices, output, count);
}}
)", kThreads, kItemsPerThread, kThreads);
    }
    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_UNIFIED_CHECK(launch_kernel(kernel, config, args.packed, args.indices, args.output, args.count));
    }
};

static void dequantize_mxfp4_fp8(const std::pair<torch::Tensor, torch::Tensor>& rhs,
                                const torch::Tensor& output) {
    TORCH_CHECK(output.is_cuda(), "MXFP4 decoding requires PPU tensors");
    c10::DeviceGuard guard(output.device());
    TORCH_CHECK(is_ppu1v5_device(), "MXFP4 decoding requires PPU1.5");
    const auto& [packed, indices] = rhs;
    TORCH_CHECK((output.dim() == 2 || output.dim() == 3) && packed.dim() == output.dim() &&
                    indices.dim() == output.dim(), "MXFP4 decode expects dense or grouped NT weights");
    const int64_t n = output.size(-2), k = output.size(-1);
    auto packed_shape = output.sizes().vec(), index_shape = output.sizes().vec();
    packed_shape.back() = k / 2;
    index_shape.back() = k / 32;
    TORCH_CHECK(output.numel() > 0 && n % 128 == 0 && k % 128 == 0 &&
                    packed.sizes() == packed_shape && indices.sizes() == index_shape,
                "MXFP4 decode requires [E,N,K/2] uint8 weights and prepared [E,N,K/32] indices; "
                "N and K must be divisible by 128 (omit E for dense)");
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8 && indices.scalar_type() == torch::kUInt8 &&
                    output.scalar_type() == torch::kFloat8_e4m3fn,
                "MXFP4 decode requires W4A8_MXFP4_FP8 prepared weights/indices and E4M3 output");
    for (const auto& tensor : {packed, indices, output})
        TORCH_CHECK(tensor.device() == output.device() && tensor.is_contiguous(),
                    "MXFP4 decode tensors must be contiguous on one PPU");
    TORCH_CHECK(output.numel() / 32 <= std::numeric_limits<int32_t>::max(),
                "MXFP4 decode supports at most INT32_MAX K32 records");
    const auto begin = reinterpret_cast<uintptr_t>(output.data_ptr());
    TORCH_CHECK(begin % 16 == 0 && reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0,
                "MXFP4 decode requires 16-byte aligned weights and output");
    for (const auto& input : {packed, indices}) {
        const auto address = reinterpret_cast<uintptr_t>(input.data_ptr());
        TORCH_CHECK(begin + output.nbytes() <= address || address + input.nbytes() <= begin,
                    "MXFP4 decode output must not overlap its inputs");
    }
    const uint32_t count = output.numel() / 32;
    using Runtime = Mxfp4DecodeRuntime;
    Mxfp4DecodeRuntime::Args args{packed.data_ptr(), indices.data_ptr(), output.data_ptr(), count,
        {dim3(ceil_div(count, uint32_t(Runtime::kItemsPerThread * Runtime::kThreads))), dim3(Runtime::kThreads), 0}};
    const auto& runtime = compiler->build("dequantize_mxfp4_fp8", Runtime::generate(args));
    Runtime::launch(runtime, args);
}

} // namespace deep_gemm::w4fa8
