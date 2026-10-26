# UT Functional Test
Use `run_deep_gemm.py` with --format to specify input parameters.
## Usage
run_deep_gemm.py [-h] [--format FORMAT] [--caselist CASELIST] [--case_idx CASE_IDX] [--disable_acc] [--benchmark] [--ref_backend {host,device}]

options: \
  -h, --help            show this help message and exit \
  --format FORMAT       Case cmd to describe problem size. \
  --caselist CASELIST   the folder of DG cases \
  --case_idx CASE_IDX   the line index(1~line) of case in caselist file \
  --disable_acc         if or not open accuracy check \
  --benchmark           specify if run benchmark \
  --ref_backend {host,device}
                        specify the backend used to compute ref output \
## Example
### Accuracy test: compare deepgemm with pytorch cpu implementation
[recommended for cmodel environment] \
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12 --ref_backend=host

### Accuracy test: compare deepgemm with pytorch device kernel implementation
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12

### Run deepgemm kernel only without correctness check
python ./run_deep_gemm.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12 --disable_acc


# UT Performance Test
Use `run_deep_gemm_perf.py` with --format to specify input parameters and --mode to specify mode.
## Usage
run_deep_gemm_perf.py [-h] [--caselist CASELIST] [--case_idx CASE_IDX] [--format FORMAT] [--output OUTPUT] [--mode {metrics,full,show_log,umd_perf}]
                             [--device DEVICE] [--disable_acc] [--ref_backend {host,device}]

options: \
  -h, --help            show this help message and exit \
  --caselist CASELIST   the folder of DG cases \
  --case_idx CASE_IDX   the line index(1~line) of case in caselist file \
  --format FORMAT       Case cmd to describe problem size. \
  --output OUTPUT       the output storing cycles of DG cases \
  --mode {metrics,full,show_log,umd_perf}
                        run perf mode \
  --device DEVICE       devices index to run cases, 0 means gpu0. 0,3 means gpu0,1,2,3 \
  --disable_acc         if or not open accuracy check \
  --ref_backend {host,device}
                        specify the backend used to compute ref output \

## Example
### Run a single case with parameters specified by format
python ./run_deep_gemm_perf.py --format=GroupedMasked,data_type:int8,groups:16,m:4096,n:4096,k:7168,em:12,distribution:[45,200,30,18,13,21,31,3,5,17,17,16,18,15,18,13]

### Run a batch of cases (accuracy + performance by default)
python ./run_deep_gemm_perf.py --caselist=./deepgemm.caselist

# Checkin Test
## Test fp8
python ./test_fp8_core.py

## Test int8 and bf16
python ./test_core.py

test_core.py options:
  -h, --help            show this help message and exit
  --func {DenseGemm,GroupedContiguous,GroupedMasked,DenseGemm}  target test func
# Routed MoE correctness and performance tests

MegaMoE supports W4A4 on ZW-M890P. Following the upstream SM90 test/benchmark
split, `test_mega_moe.py` contains public-entry correctness and regression tests;
`bench_mega_moe.py` contains full-pipeline timing and optional DeepEP comparisons.
The benchmark reuses the correctness runner's case table and input/reference
helpers. Both use the public complete call.
Other architecture pytest runs skip these tests; CLI use reports an
unsupported-device error.

| Case | EP | H | I | Experts | Top-k | Activation | Capacity |
|---|---|---:|---:|---:|---:|---|---:|
| `deepseek_v4_pro` | 8 | 7168 | 3072 | 384 | 6 | SwiGLU, limit 10 | 8192 |
| `deepseek_v4_flash` | 4 | 4096 | 2048 | 256 | 6 | SwiGLU, limit 10 | 4096 |
| `glm52` | 8 | 6144 | 2048 | 256 | 8 | SwiGLU | 4096 |
| `kimi_k3` | 8,16 | 3584 | 3072 | 896 | 16 | SiTU | 4096 |
| `qwen38_2_4t` | 8,16 | 8192 | 2048 | 512 | 10 | SwiGLU | 4096 |
| `mimo_v26_pro` | 8 | 6144 | 2048 | 384 | 8 | SwiGLU | 8192 |

I is the activated intermediate width; GEMM1 has 2I output rows. Primary model
tokens/rank are 1024, 2048 and 4096; MiMo also includes 8192. CLI overrides can
select 32 tokens or other counts within capacity. `--num-tokens` selects
actual tokens and `--num-max-tokens-per-rank` selects workspace capacity;
`--tokens` and `--capacity` remain CLI aliases. `--routing` supports
`round_robin` (default), `random` and `skewed`. Use random routing explicitly
for comparisons with random-route benchmark reports.
`--fast-math 1` is the default and passes `fast_math=True` to the public call;
`--fast-math 0` selects the regular activation math path. The three smoke
activations cover both modes, including graph replay and master/replica weights.

```bash
# Three activations plus independent G1/G2 tile configurations, two idle devices.
torchrun --standalone --nproc-per-node=2 -m pytest -q -s -k smoke \
  tests/test_mega_moe.py

# Correctness for DeepSeek V4 Pro.
torchrun --standalone --nproc-per-node=8 tests/test_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 32 1024 4096 --routing random \
  --reference-tokens 8

# Full-pipeline graph latency without a baseline.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 32 1024 4096 --routing random \
  --reference-tokens 8 --warmup 10 --iterations 100 --repeats 7

# Prefill against DeepEP normal, with eager timing for both paths.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 1024 2048 4096 --routing random \
  --reference-tokens 8 --baseline normal --warmup 10 --iterations 100 --repeats 7

# Decode against DeepEP low latency, with graph timing for both paths.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case deepseek_v4_pro --num-tokens 32 64 128 --routing random \
  --reference-tokens 8 --baseline low_latency --warmup 10 --iterations 100 --repeats 7

# SiTU correctness and performance with separate master/replica weights.
torchrun --standalone --nproc-per-node=8 tests/bench_mega_moe.py \
  --case kimi_k3 --num-tokens 1024 2048 4096 --routing random --replicas \
  --reference-tokens 8
```

EP16 uses two nodes of eight devices: set `--nnodes=2`, `--node-rank=0/1`,
`--master-addr` and `--master-port` consistently. Enable the appropriate
symmetric-memory fabric transport and use idle devices on every rank.
For single-node containers whose hostname is not resolvable, replace
`--standalone` with `--nnodes=1 --node-rank=0 --master-addr=127.0.0.1
--master-port=29500` (choose an unused port).
CI allocates two ZW-M890P devices for the EP2 suite; the existing ordinary
GEMM suites retain their original coverage.

Smoke SwiGLU, Step and SiTU cases use 1, 0, 33, 129 and 8 tokens. The 129-token
case crosses a 128-row tile boundary. Every smoke token is checked against an
independent quantize/dequantize and expert-matmul reference. References apply
original routing weights after GEMM2. The shared `deep_gemm.testing.calc_diff`
measures normalized squared error, mathematically equivalent to
`sum((actual-reference)^2) / sum(actual^2+reference^2)`, and returns zero when
both tensors are all zero; W4A4 requires error below 0.003.
Relative L2 is also reported. `--reference-tokens N` samples N tokens per rank
for large-model mathematical reference checks; omit it to check all tokens.

The same runner validates direct `SymmBuffer` input-view writes, output-derived
token counts, output offset views, independent G1/G2 tile sizes,
empty ranks, invalid IDs, zero weights with NaN-filled unwritten combine slots,
and master/replica partitions. Consecutive calls and graph replays change
inputs, routes and weights and compare against fresh references, including
hot experts, cross-rank destinations and fully inactive routes. It checks
counter recycling and workspace scratch reuse without caller-managed offsets.
Unsupported MMA type, activation, tile, dtype and shape combinations are
rejected before entering the distributed compute protocol. `--edge-cases`
enables the CLI's empty-rank and masked-route variants.

Smoke tests also check SM90-style positional calls and reject incompatible
recipes and malformed receive-statistics tensors. INT32 cumulative statistics
are compared against independent all-rank routing histograms, starting from a
nonzero accumulator. Tests cover master/replica experts, empty ranks, invalid
and zero-weight routes, compile-only mode, repeated calls and graph replay
with changing routes. Capture must not increment statistics; each replay must.

Timing starts with the same BF16 inputs and includes required input preparation,
quantization, dispatch, GEMM1/activation/output quantization, metadata,
GEMM2/scatter, state recycling and combine. One-time workspace allocation,
weight transforms, JIT, reference checks and harness barriers are outside the
timed region. DeepEP normal rebuilds its dispatch layout and expert permutation
on every forward; these operations and any required host/device transfers are
included. Original routing weights are applied once, after GEMM2, in both paths.
Both backends use the same raw quantized weights, inputs, routing IDs and weights.

Each sample is the slowest rank's mean latency; JSON reports contain all samples
and their median/min/max. `baseline.speedup` is DeepEP latency divided by MegaMoE
latency. `--baseline none` (default) and `--baseline low_latency` default to graph
timing. DeepEP normal returns CPU expert counts and requires eager execution;
`--baseline normal` therefore selects eager for both paths and rejects
`--mode graph`. Eager timing includes host launch gaps. Graph and eager results
must not be mixed in the same speedup calculation.

DeepEP and Triton are optional and are imported only for a requested baseline.
If either is absent on any rank, all ranks skip the comparison and report the
reason while still benchmarking MegaMoE. An installed backend's import or
execution errors fail the run. All three W4A4 activations are implemented:

| Activation | DeepEP expert computation |
|---|---|
| SwiGLU | Grouped GEMM1 with fused activation and MXFP4 quantization |
| Step | Grouped GEMM1 with fused Step activation and MXFP4 quantization |
| SiTU | Grouped GEMM1, then standalone `situ_and_mul` with MXFP4 quantization |

The standalone SiTU kernel is included in baseline timing; the ordinary grouped
GEMM currently has no fused SiTU epilogue. JSON records `activation_impl` so this
difference is visible. MegaMoE uses the configured case capacity; DeepEP LL sizes
its per-expert receive storage for the largest requested token count, which is
also recorded as `baseline.num_max_tokens_per_rank`.
DeepEP has its own shape restrictions: the installed LL backend may reject the
smoke cases' hidden width of 512. Use a supported model case for LL comparisons;
backend errors are reported rather than changing the requested shape.

Correctness and changed-route graph replay run before timing by default, using
the existing W4A4 numerical threshold. DeepEP is also checked against the same
independent reference. `bench_mega_moe.py --no-check` skips correctness checks.
The old `test_mega_moe.py ... --benchmark` command forwards to the benchmark
entry and preserves the existing token/capacity aliases and timing arguments.

Measured model profiles live in `csrc/jit_kernels/heuristics/mega_moe_tuning.hpp`.
They match exact shape, EP, activation and actual token count. Unknown cases
use the default heuristic. Workspace `gemm1_config` / `gemm2_config` tiles
override the relevant profile stage. `MEGA_MOE_PULL_SMS` remains available for
pull CTA experiments; normal benchmark commands require no tuning overrides.
The public calling example and workspace reuse contract are in the main README.
