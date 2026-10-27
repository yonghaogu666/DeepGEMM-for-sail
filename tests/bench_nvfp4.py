"""Compare fused W4A8, per-call FP4 decode + FP8, and FP8 on the same weights.

B counts input tokens per EP rank before routing. Synthetic balanced routing
gives topk * B local rows across 384 / EP experts. This measures one local
GEMM, excluding routing, communication, quantization, packing and allocation.

Run: python tests/bench_nvfp4.py --ep 64 --stage 1 --device 0 --output nvfp4.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import deep_gemm as dg


def measure(fn, output=None, baseline=None):
    for _ in range(12):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = []
    for _ in range(15):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    if output is not None:
        output.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        error = relative_l2(output, baseline)
        assert error < 0.006, ('graph refresh', error)
    return {'us': statistics.median(samples), 'samples_us': samples}


def relative_l2(output, baseline):
    return ((output.float() - baseline.float()).norm()
            / baseline.float().norm().clamp_min(1e-20)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ep', type=int, choices=[32, 64], required=True)
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--device', type=int, default=0, help='PyTorch device index')
    parser.add_argument('--batches', type=int, nargs='+', default=[2 ** i for i in range(13)],
                        help='Pre-routing tokens per rank (default: powers of two from 1 to 4096)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or any(b <= 0 for b in args.batches):
        parser.error('device must be nonnegative and batch sizes must be positive')
    torch.cuda.set_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    experts, topk = 384, 4
    n, k = (10240, 7168) if args.stage == 1 else (7168, 5120)
    groups = experts // args.ep
    torch.manual_seed(42)
    checkpoint = torch.randint(256, (groups, n, k // 2), device='cuda', dtype=torch.uint8)
    native_scales = torch.rand((groups, n, k // 16), device='cuda').add(0.125).to(torch.float8_e4m3fn)
    outer = torch.rand((groups, n // 128, k // 128), device='cuda').add(0.25)
    fused_rhs = dg.preprocess_weights_and_scales(
        dg.PreprocessMode.W4A8_NVFP4, False, checkpoint, native_scales)
    prefill_rhs = dg.preprocess_weights_and_scales(
        dg.PreprocessMode.W4A8_NVFP4_FP8, False, checkpoint, native_scales)
    del checkpoint, native_scales
    workspace = torch.empty((groups, n, k), device='cuda', dtype=torch.float8_e4m3fn)

    def decode():
        dg.dequantize_nvfp4_fp8(prefill_rhs, workspace)

    decode()
    receipt = {
        'format': 'nvfp4',
        'ep': args.ep,
        'stage': args.stage,
        'groups': groups,
        'topk': topk,
        'n': n,
        'k': k,
        'torch': torch.__version__,
        'device': torch.cuda.get_device_name(),
        'device_index': torch.cuda.current_device(),
        'deep_gemm': dg.__file__,
        'workspace_bytes': workspace.nbytes,
        'scope': 'one grouped GEMM; B is pre-routing tokens per rank; balanced local rows=topk*B',
        'timing': 'CUDA graph; 12 warmups, 20 inner calls, 15 rounds per method order; '
                  'forward/reverse GEMM orders, 30 pooled samples, median microseconds',
        'baseline': 'FP8 uses identical decoded weights and outer scales',
        'data': 'synthetic FP4 payloads, K16 E4M3 scales in [0.125,1.125], and FP8 activations',
        'activation_scale': '1x128 FP32',
        'outer_weight_scale': '128x128 FP32',
        'weight_global_scale': 1.0,
        'decode': measure(decode),
        'rows': [],
        'summary': [],
    }
    run_fused = dg.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad
    for b in args.batches:
        m = topk * b
        counts = torch.full((groups,), m // groups, device='cuda', dtype=torch.int32)
        counts[: m % groups] += 1
        ids = torch.repeat_interleave(torch.arange(groups, device='cuda', dtype=torch.int32), counts)
        a = torch.randn((m, k), device='cuda').to(torch.float8_e4m3fn)
        sa = torch.rand((m, k // 128), device='cuda').add(0.25)
        sa_fp8 = dg.get_col_major_tma_aligned_tensor(sa)
        output = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

        def fp8():
            dg.m_grouped_fp8_gemm_nt_nopad((a, sa_fp8), (workspace, outer), output, ids, counts)

        def prefill():
            run_fused(
                (a, sa_fp8),
                prefill_rhs,
                output,
                ids,
                m_rows=counts,
                lhs_scale_group_size=128,
                rhs_scales=outer,
                rhs_scale_block_size=128,
                execution='dequantize',
                fp8_workspace=workspace,
            )

        def fused():
            run_fused(
                (a, sa),
                fused_rhs,
                output,
                ids,
                m_rows=counts,
                lhs_scale_group_size=128,
                rhs_scales=outer,
                rhs_scale_block_size=128,
                execution='fused',
            )

        fp8()
        baseline = output.clone()
        methods = [('fp8', fp8), ('decode_fp8', prefill), ('fused', fused)]
        samples = {method: [] for method, _ in methods}
        errors = {method: [] for method, _ in methods}
        for pass_index in range(2):
            for method, fn in methods if pass_index == 0 else reversed(methods):
                fn()
                error = relative_l2(output, baseline)
                assert error < 0.006, (b, method, error)
                row = {'B': b, 'local_rows': m, 'method': method, 'pass': pass_index,
                       'relative_l2_vs_fp8': error, **measure(fn, output, baseline)}
                receipt['rows'].append(row)
                samples[method].extend(row['samples_us'])
                errors[method].append(error)
        fp8_us = statistics.median(samples['fp8'])
        for method, _ in methods:
            us = statistics.median(samples[method])
            summary = {'B': b, 'local_rows': m, 'expert_rows': counts.tolist(), 'method': method,
                       'us': us, 'tflops': 2 * m * n * k / (us * 1e6),
                       'fp8_over_method': fp8_us / us,
                       'relative_l2_vs_fp8': max(errors[method])}
            receipt['summary'].append(summary)
            print(json.dumps(summary), flush=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
