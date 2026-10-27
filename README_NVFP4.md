# NVFP4 W4A8 GEMMs (M890P)

Build the library using the main [Quick Start](README.md#quick-start).

## Interfaces and scale layouts

Use the existing `m_grouped_gemm_fp8_fp4_bf16_nt_{nopad,masked,fused}`
interfaces and `PreprocessMode.W4A8_MXFP4_MMA` for MXFP4 weights.
NVFP4 uses independent K16 E4M3 scales and a separate prepared int32 layout.

An original NVFP4 weight is `q_e2m1 * native_k16_scale * global_scale`.
The K16 scale is E4M3FN; the global scale is FP32. The optional FP8
`rhs_scales` argument is a separate restoring multiplier for the kernel's
weight representation. See [offline scale absorption](#offline-fp8-scale-absorption)
for converting original checkpoint scales to that representation.

`gemm_fp8_nvfp4_bf16_nt` and `m_grouped_gemm_fp8_nvfp4_bf16_nt_{nopad,masked,fused}`
consume `(activation, activation_scale)` and `(prepared_weight, native_scale)`
pairs and produce BF16 output. Activations are E4M3FN. K must be divisible
by 32 and N by 64. Prepare checkpoint uint8 `[E,N,K/2]` weights (low nibble
first along K) and E4M3FN `[E,N,K/16]` scales through the shared
`preprocess_weights_and_scales(PreprocessMode.W4A8_NVFP4, False, weight, scale)`
entry point. Dense GEMMs also accept prepared buffers with E=1 removed.
Prepared buffers are not interchangeable with the MXFP4 MMA layout.

Like the upstream MXFP4 W4A8 interfaces, `rhs[1]` holds the prepared native
FP4 scales. The additional FP8 restoring scale and NVFP4 global scale have
separate keyword arguments:

| Argument | Contents | Storage dtype |
| :--- | :--- | :--- |
| `rhs[0]` | Prepared packed NVFP4 codes | int32 |
| `rhs[1]` | Prepared K16 scales | uint8 bytes encoding E4M3FN |
| `rhs_scales` | Optional FP8 restoring scales | FP32 |
| `weight_global_scale` | Original NVFP4 global scale | FP32 |

The activation scale is FP32 `[M,1]` for per-token scaling
(`lhs_scale_group_size=0`),
or `[M,ceil(K/G)]` with `lhs_scale_group_size=G` for block scaling
(G=16 or a positive multiple of 32). Masked buffers add E as the first
dimension. `(activation, None)` selects unit per-token scaling in fused
execution. An optional `weight_global_scale` is an FP32 scalar or `[E]`.
With `rhs_scale_block_size=0`, optional per-channel `rhs_scales` are
`[N]`/`[N,1]` for dense and `[E,N]`/`[E,N,1]` for grouped. All scales are
multiplicative.

For the FP8 blockwise path, pass `rhs_scale_block_size=128` with FP32
`rhs_scales` shaped `[ceil(N/128),ceil(K/128)]` for dense or
`[E,ceil(N/128),ceil(K/128)]` for grouped. E2M1 times its native K16
E4M3 scale is rounded to E4M3 (nearest-even, finite saturation) before MMA.
Outer activation/weight block scales apply to FP32 partial sums. Without
outer weight block scales, fused execution retains the native K16 scaling
without FP8 weight rounding. All restoring scales apply before BF16 output.

Nopad rows must be sorted by expert; masked calls preserve invalid rows.
Fused routing uses `moe_align_block_size` and a matching GEMM configuration,
as described under [routing and configurations](#routing-and-configurations).

For prefill, prepare checkpoint uint8 `[E,N,K/2]` weights and E4M3FN
`[E,N,K/16]` scales using
`preprocess_weights_and_scales(PreprocessMode.W4A8_NVFP4_FP8, False, ...)`.
Pass `execution='dequantize'` and a caller-owned contiguous E4M3FN
`fp8_workspace` shaped `[E,N,K]` (or `[N,K]` for dense). Each call decodes
the current weights and then invokes the existing FP8 GEMM on the current
stream. Explicit FP32 activation and 128x128 outer weight scales are
required; activations may be per-token
or K128. `execution='fused'` remains the default for online decoding.
`dequantize_nvfp4_fp8(rhs, workspace)` also exposes the FP8 decoding step;
outer weight and global scales are applied by the subsequent GEMM.

## Runnable dense and nopad examples

Run these blocks in order on PPU1.5 after building the library. The example
uses FP8 activations with FP32 1x128 scales, small native K16 scales and unit
FP32 128x128 outer weight scales. For a checkpoint requiring FP8 normalization,
use normalized K16 scales and the corresponding restoring outer scales from
the [offline conversion](#offline-fp8-scale-absorption).

```python
import torch
import deep_gemm as dg

torch.manual_seed(42)
e, m, n, k = 3, 35, 256, 256
codes = torch.randint(16, (e, n, k), device='cuda', dtype=torch.uint8)
checkpoint_weight = codes[..., ::2] | (codes[..., 1::2] << 4)
native_k16_scale = (torch.rand(e, n, k // 16, device='cuda') + 0.25).to(torch.float8_e4m3fn)
a = torch.randn(m, k, device='cuda').to(torch.float8_e4m3fn)
a_scale = torch.rand(m, k // 128, device='cuda', dtype=torch.float32) + 0.25
weight_block_scale = torch.ones(e, n // 128, k // 128, device='cuda', dtype=torch.float32)
weight_global_scale = torch.tensor([0.5, 0.75, 1.25], device='cuda', dtype=torch.float32)

# Prepare both layouts once, outside the GEMM call. Prepare only the layout
# required by the execution mode used by the caller.
rhs = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_NVFP4, False, checkpoint_weight, native_k16_scale)
rhs_fp8 = dg.preprocess_weights_and_scales(
    dg.PreprocessMode.W4A8_NVFP4_FP8, False, checkpoint_weight, native_k16_scale)
workspace = torch.empty((e, n, k), device='cuda', dtype=torch.float8_e4m3fn)
```

Dense GEMM uses one expert's weights. Select the execution mode explicitly
according to the caller's prefill/decode phase; token count does not select it.

```python
out_dense = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

# Decode: expand weights inside GEMM, without a full FP8 weight workspace.
dg.gemm_fp8_nvfp4_bf16_nt(
    (a, a_scale), tuple(x[0] for x in rhs), out_dense,
    lhs_scale_group_size=128, rhs_scales=weight_block_scale[0],
    rhs_scale_block_size=128, weight_global_scale=weight_global_scale[0],
    execution='fused')

# Prefill: decode all current weights, then invoke the existing FP8 GEMM.
dg.gemm_fp8_nvfp4_bf16_nt(
    (a, a_scale), tuple(x[0] for x in rhs_fp8), out_dense,
    lhs_scale_group_size=128, rhs_scales=weight_block_scale[0],
    rhs_scale_block_size=128, weight_global_scale=weight_global_scale[0],
    execution='dequantize', fp8_workspace=workspace[0])
```

For nopad grouped GEMM, activation rows must already be sorted by expert.
`m_indices` is the expert ID of each row and `m_rows` contains per-expert
counts. This example includes an empty expert.

```python
m_rows = torch.tensor([17, 0, 18], device='cuda', dtype=torch.int32)
m_indices = torch.repeat_interleave(
    torch.arange(e, device='cuda', dtype=torch.int32), m_rows)
out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

dg.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
    (a, a_scale), rhs, out, m_indices, m_rows=m_rows,
    lhs_scale_group_size=128, rhs_scales=weight_block_scale,
    rhs_scale_block_size=128, weight_global_scale=weight_global_scale,
    execution='fused')

dg.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
    (a, a_scale), rhs_fp8, out, m_indices, m_rows=m_rows,
    lhs_scale_group_size=128, rhs_scales=weight_block_scale,
    rhs_scale_block_size=128, weight_global_scale=weight_global_scale,
    execution='dequantize', fp8_workspace=workspace)
```

The FP8 workspace occupies `E*N*K` bytes, or `N*K` bytes for dense. Allocate
it outside the timed region and keep it separate from inputs, outputs and
routing metadata. It is overwritten on each `execution='dequantize'` call;
the API does not retain a cross-call FP8 weight cache. The caller controls
its lifetime. Retaining both prepared weight layouts is optional.

## Offline FP8 scale absorption

The blockwise execution paths decode `q_e2m1 * native_k16_scale` to FP8
before applying `rhs_scales` and `weight_global_scale`. They do not divide
by `rhs_scales` during decoding. To normalize the FP8 values, absorb that
division into the K16 scales offline, before checkpoint preprocessing:

```text
normalized_k16_scale = E4M3(native_k16_scale / fp8_block_scale)
decoded_fp8 = E4M3(q_e2m1 * normalized_k16_scale)
restored_weight = decoded_fp8 * fp8_block_scale * global_scale
```

Pass `normalized_k16_scale` to `preprocess_weights_and_scales`,
`fp8_block_scale` as `rhs_scales`, and the original global scale as
`weight_global_scale`. Both `execution='fused'` and `execution='dequantize'`
use these same scale arguments; no kernel or API changes are needed.
The normalized K16 scales and outer scales together represent the weights,
so retain both rather than treating the normalized K16 scales as the
original checkpoint scales.

[tests/test_nvfp4.py](tests/test_nvfp4.py), helper `_absorb_fp8_scale`,
demonstrates this conversion for N/K divisible by 128. It chooses a
power-of-two scale per 128x128 weight block, keeping both the normalized K16 scales and their
decoded values within E4M3 range. This avoids additional mantissa rounding
when rescaling normal E4M3 values, although small K16 scales can underflow.
It is not a lossless conversion in general.

## Routing and configurations

Keep the default configuration for dense, nopad and masked calls.
For fused routing with `execution='fused'`, select a configuration with
`get_gemm_fp8_nvfp4_config('MGroupedFused', expected_m, n, k,
lhs_scale_group_size=G, rhs_scale_block_size=128)` and pass it to both
`moe_align_block_size` and GEMM. The selector also accepts `Normal`,
`MGroupedNoPad` and `MGroupedMasked`.

For fused routing with `execution='dequantize'`, pass the FP8 workspace to
`moe_align_block_size` and use its returned seven-entry FP8 configuration.
This path requires N/K divisible by 128 and `block_k=128`. The fused NVFP4
execution uses an eight-entry configuration; the two are not interchangeable.
The API suffix `..._fused` denotes routing gather, independently of the
`execution` option. The tests include examples of both routing paths.

## Tests and further API examples

NVFP4 tests and the offline scale example are in
[tests/test_nvfp4.py](tests/test_nvfp4.py). The upstream MXFP4 tests remain
in [tests/test_w4a8.py](tests/test_w4a8.py). Run from the repository root
with the built checkout on `PYTHONPATH`:

```bash
# All three NVFP4 tests, including offline scale comparisons.
python -m pytest tests/test_nvfp4.py -q -s

# One test, e.g. offline scale absorption in both execution modes.
python -m pytest tests/test_nvfp4.py::test_nvfp4_offline_scale -q -s

# Standalone runner for the same three NVFP4 tests.
python tests/test_nvfp4.py
```

The decode/prefill tests cover dense, nopad, masked and fused-routing GEMMs with
batch sizes 1, 17, 128, 257 and 4096. Batch denotes activation rows, or the
per-expert buffer capacity for masked GEMMs; fused routing uses top-k=2.
They include empty experts, partial N/K tiles, activation and weight scales,
invalid K-warp reduction configurations, and one graph replay with changed
weights/scales per execution mode.
References are independently constructed with PyTorch FP32 operations.

| Execution / coverage | Test function |
| :--- | :--- |
| Fused decoding for all four GEMM interfaces; one shared-decode tile | `test_nvfp4_decode` |
| Per-call decode-to-FP8 for all four GEMM interfaces; exact FP8 workspace checks | `test_nvfp4_prefill` |
| Offline FP8 scale absorption versus direct casts; original NVFP4 and converted FP8 references | `test_nvfp4_offline_scale` |

## Performance benchmark

[tests/bench_nvfp4.py](tests/bench_nvfp4.py) compares fused W4A8,
per-call decode plus FP8 GEMM, and FP8 GEMM using these model dimensions:

| Stage | N | K |
| :--- | ---: | ---: |
| `--stage 1`: gate/up | 10240 | 7168 |
| `--stage 2`: down | 7168 | 5120 |

There are 384 experts and top-k=4. `--ep 32` selects 12 local experts;
`--ep 64` selects 6. B is the number of input tokens per rank before
routing. Balanced routing gives `M=4*B` local GEMM rows, distributed as
evenly as possible across those experts. The benchmark runs on one device
and excludes routing, communication, quantization, packing and allocation.
Weights, scales and activations are synthetic; model checkpoint data is
not loaded. Activations use FP32 1x128 scales, outer weights use FP32
128x128 scales, and the global weight scale is one.

```bash
python tests/bench_nvfp4.py --device 0 --ep 64 --stage 1 \
    --output results/nvfp4-ep64-up.json
```

The default B values are powers of two from 1 to 4096. Use
`--batches 1 128 4096` to select a subset, and select the other EP/stage
values to measure all four configurations.

The FP8 baseline uses the identical decoded weights and restoring scales.
Each path is checked against that baseline before timing and after Graph
replay. This checks agreement with FP8 computation; original-NVFP4
conversion accuracy is covered by the separate offline-scale test.

Timing uses CUDA Graphs with 12 warmups, 20 calls per replay and 15 rounds
in each of forward/reverse method orders. JSON `rows` retains both passes
and their samples; `summary` pools the 30 samples per B/method and reports
median microseconds, TFLOPS and `fp8_over_method` (`FP8 time / method time`,
above one means that method is faster). The standalone decoder is also
timed separately.

## Accuracy references

The offline-scale test uses three synthetic K16 scale distributions, each
rounded to E4M3. These names describe scale values, not batch sizes or
statistics from a model checkpoint:

| Profile | Original K16 scale range | Sampling |
| :--- | :--- | :--- |
| `small` | 0.25 to 1.25 | Uniform |
| `large` | 128 to 448 | Uniform |
| `wide` | 2^-9 to 448 | Log-uniform |

The `large` and `wide` cases also set each row's first K16 scale to 448
and its first FP4 value to 6 to exercise FP8 saturation. Both conversions
use the same FP4 codes and global scales. Dense/nopad checks use three
256x256 expert weights and batch sizes 1, 128 and 4096.

For the same activation values and scales, define:

- `Y_nvfp4`: FP32 GEMM using the original NVFP4 weights decoded in FP32,
  including the original K16 and global scales.
- `Y_fp8`: FP32 GEMM using the converted FP8 weights and their restoring scales.
- `Y_kernel`: the actual BF16 output from either GEMM execution mode.

Each relative L2 error is `norm(actual - reference) / norm(reference)`;
multiply by 100 for a percentage. The printed fields mean:

| Field | Comparison | What it measures |
| :--- | :--- | :--- |
| `weight_vs_nvfp4_rel_l2` | Restored FP8 weights vs original NVFP4 weights | Weight conversion |
| `fp8_reference_vs_nvfp4_rel_l2` | `Y_fp8` vs `Y_nvfp4` | Conversion effect on GEMM output |
| `fused_vs_nvfp4_rel_l2`, `dequantize_vs_nvfp4_rel_l2` | `Y_kernel` vs `Y_nvfp4` | Conversion plus GEMM/output rounding |
| `*_vs_fp8_reference_rel_l2` | `Y_kernel` vs `Y_fp8` | GEMM arithmetic and BF16 output rounding |
| `fused_vs_dequantize_rel_l2` | Fused output vs decode-to-FP8 output | Agreement between execution modes |

The total and arithmetic errors use different references and do not add
directly. They measure conversion and computation after NVFP4 quantization;
model accuracy against the weights before NVFP4 quantization is not evaluated.
