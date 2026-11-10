#pragma once
#include <cstdint>
namespace deep_gemm::w4fa8 {
enum class FragmentPolicy { FullK, RollingK32 };
enum class ALoadPolicy { Bulk, Ring2 };

constexpr bool uses_ring_a(int warp_m) {
  return warp_m >= 80 && warp_m <= 192 && warp_m % 16 == 0;
}

template <class Storage, int Rows> struct RowScaleStorage : Storage {
  alignas(128) float row_scales[2][Rows];
};
template <class Storage> struct RowScaleStorage<Storage, 0> : Storage {};
} // namespace deep_gemm::w4fa8

namespace deep_gemm::w4fa8_streamed {
inline constexpr int kBlockM = 256;
// Stage K128 scales alongside the operand pipeline, independent of total K.
// Host launch sizing and device storage use the same capacity decision.
constexpr int group_scale_pipeline_elements(int rows, int stages, int operand_bytes, bool grouped) {
  const int64_t elements = int64_t(rows) * stages;
  return grouped && operand_bytes + elements * sizeof(float) <= 256 * 1024 ? int(elements) : 0;
}

template <class Storage, int Elements> struct GroupScaleStorage : Storage {
  alignas(128) float group_scales[Elements];
};
template <class Storage> struct GroupScaleStorage<Storage, 0> : Storage {};

struct GroupedRowRangeArguments {
  uint32_t shape_m;
  int32_t* grouped_layout;
  const int32_t* row_range;
};
struct GatherParams {
  const uint8_t *a;
  const float *a_scale;
  uint8_t *grouped_a;
  float *grouped_scale;
  const int32_t *experts;
  const int32_t *tokens;
  const int32_t *aligned;
  int num_tokens;
};
struct DecodeParams {
  const uint8_t* packed;
  const uint8_t* scales;
  uint8_t* output;
  int n_begin;
  const int32_t* all_rows;
  int group_start;
};
} // namespace deep_gemm::w4fa8_streamed
