#pragma once
#include <hggc_runtime.h>
#include <stdint.h>

namespace deep_gemm::w4fa8 {

// Byte i is E4M3_RNE(E2M1_magnitude[i] * 2^(index-13)). Index zero
// also represents smaller exponents, which all round to zero. Offline
// normalization bounds index to [0,19]; signs are restored separately.
static __device__ __constant__ uint64_t kMxfp4E4m3ScaleLut[20] = {
    0x0000000000000000ull,  // 2^-13
    0x0100000000000000ull,  // 2^-12
    0x0201010000000000ull,  // 2^-11
    0x0302020101000000ull,  // 2^-10
    0x0604030202010000ull,  // 2^-9
    0x0c08060403020100ull,  // 2^-8
    0x14100c0806040200ull,  // 2^-7
    0x1c1814100c080400ull,  // 2^-6
    0x24201c1814100800ull,  // 2^-5
    0x2c2824201c181000ull,  // 2^-4
    0x34302c2824201800ull,  // 2^-3
    0x3c3834302c282000ull,  // 2^-2
    0x44403c3834302800ull,  // 2^-1
    0x4c4844403c383000ull,  // 2^0
    0x54504c4844403800ull,  // 2^1
    0x5c5854504c484000ull,  // 2^2
    0x64605c5854504800ull,  // 2^3
    0x6c6864605c585000ull,  // 2^4
    0x74706c6864605800ull,  // 2^5
    0x7c7874706c686000ull,  // 2^6
};

__device__ __forceinline__ uint4 decode_fp4x16(uint2 bits, uint32_t lo, uint32_t hi) {
    // Duplicate each natural packed byte's two sign bits into the destination
    // byte MSBs. PRMT selects bytes from (word<<4, word), without an offline
    // packed-weight copy. The LUT selectors consume the natural K nibbles.
    const uint32_t x_shift = bits.x << 4, y_shift = bits.y << 4;
    return make_uint4(
        __byte_perm(lo, hi, bits.x & 0x7777u) | (__byte_perm(x_shift, bits.x, 0x5140u) & 0x80808080u),
        __byte_perm(lo, hi, (bits.x >> 16) & 0x7777u) | (__byte_perm(x_shift, bits.x, 0x7362u) & 0x80808080u),
        __byte_perm(lo, hi, bits.y & 0x7777u) | (__byte_perm(y_shift, bits.y, 0x5140u) & 0x80808080u),
        __byte_perm(lo, hi, (bits.y >> 16) & 0x7777u) | (__byte_perm(y_shift, bits.y, 0x7362u) & 0x80808080u));
}

// Each record is one native K32 group: 16 packed bytes -> 32 E4M3 bytes.
// All accesses are contiguous; vector stores produce the existing FP8 NT layout.
template <int Items, int Threads>
__device__ __forceinline__ void decode_mxfp4_fp8(
        const uint4* __restrict__ packed, const uint8_t* __restrict__ indices,
        uint4* __restrict__ output, uint32_t count) {
    __shared__ uint64_t table[20];
    const uint32_t tid = threadIdx.x;
    if (tid < 20) table[tid] = kMxfp4E4m3ScaleLut[tid];
    __syncthreads();
    const uint32_t base = blockIdx.x * Items * Threads;
    #pragma unroll
    for (int j = 0; j < Items; ++j) {
        const uint32_t i = base + j * Threads + tid;
        if (i < count) {
            const uint4 bits = packed[i];
            const uint8_t index = indices[i];
            const uint64_t lut = index < 20 ? table[index] : 0x7f7f7f7f7f7f7f7full;
            const uint32_t lo = uint32_t(lut), hi = uint32_t(lut >> 32);
            output[2ull * i] = decode_fp4x16(make_uint2(bits.x, bits.y), lo, hi);
            output[2ull * i + 1] = decode_fp4x16(make_uint2(bits.z, bits.w), lo, hi);
        }
    }
}

} // namespace deep_gemm::w4fa8
