"""Matched W4A8 MegaMoE / DeepEP normal benchmark on ZW-M890P.

Requires DeepEP and SGLang in addition to this DeepGEMM checkout. Example:
    torchrun --standalone --nproc-per-node=8 tests/bench_moe_w4a8_deepep.py \
        --case deepseek_v4_pro --tokens 32 1024 --output /tmp/w4a8-deepep.json

Both paths receive the same BF16 activations, raw MXFP4 weights, random routes
and routing weights. Weight transformation and router/top-k selection are
outside timing. Input quantization, route layout, communication, reordering,
both GEMMs, BF16 SwiGLU, intermediate quantization, weighting and combine are
inside. DeepEP rebuilds its dispatch handle and counts on every call.

The primary comparison is eager versus eager, including DeepEP's normal CPU
count synchronization. MegaMoE graph latency is reported separately. No cached
routes or previously dispatched activations are used in the normal baseline.
DeepEP reduces weighted local expert outputs to BF16 before cross-rank combine;
MegaMoE accumulates all weighted expert outputs at the source rank. Both are
checked against the same independent FP32-accumulation reference.
"""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import deep_gemm as dg

from moe_test_utils import (MoECase, activation, calc_diff, compile_kernels, gather,
                           make_inputs, moe_process_group, quant_w4a16_mxfp4,
                           quantize_fp8_rows)


NORMAL_STAGES = ("input_quant", "dispatch_layout", "dispatch", "count_copy_allocate",
                 "reorder", "gemm1", "activation", "activation_quant", "gemm2",
                 "weighted_gather", "combine")
MEGA_STAGES = ("pre_dispatch", "dispatch_gemm1_quant_gemm2_scatter_combine")


class Events:
    def __init__(self, names):
        self.names, self.at = names, 0
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(len(names) + 1)]
        self.events[0].record()

    def step(self, name):
        assert name == self.names[self.at]
        self.at += 1
        self.events[self.at].record()

    def values(self):
        assert self.at == len(self.names)
        return [a.elapsed_time(b) * 1000 for a, b in zip(self.events, self.events[1:])]


class DeepEPNormal:
    def __init__(self, group, hidden, w1, w2):
        from deep_ep import Buffer
        from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter_sail, ep_gather
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

        self.scatter, self.gather, self.quantize = ep_scatter_sail, ep_gather, sglang_per_token_quant_fp8
        self.w1, self.w2, self.hidden = w1, w2, hidden
        self.experts = w1[0].size(0) * group.size()
        self.dispatch_config = Buffer.get_intranode_dispatch_config(group.size(), hidden, "quant")
        self.combine_config = Buffer.get_intranode_combine_config(group.size(), hidden)
        configs = (self.dispatch_config, self.combine_config)
        self.buffer = Buffer(group,
            max(c.get_nvl_buffer_size_hint(hidden * 2, group.size()) for c in configs),
            max(c.get_rdma_buffer_size_hint(hidden * 2, group.size()) for c in configs),
            low_latency_mode=False, num_qps_per_rank=Buffer.num_sms,
            allow_mnnvl=True, explicitly_destroy=True)
        assert self.buffer.runtime.get_num_rdma_ranks() == 1, "this benchmark uses one ICN node"
        self.environment = dict(num_sms=Buffer.num_sms,
            dispatch_config=str(self.dispatch_config), combine_config=str(self.combine_config),
            local_ranks=self.buffer.local_ranks, rdma_ranks=self.buffer.runtime.get_num_rdma_ranks(),
            source_files={inspect.getsourcefile(f): hashlib.sha256(
                Path(inspect.getsourcefile(f)).read_bytes()).hexdigest()
                for f in (Buffer, ep_scatter_sail, sglang_per_token_quant_fp8)})

    def run(self, x, ids, weights, details=None):
        step = details.step if details else lambda _: None
        q, s = self.quantize(x)
        if self.w1[2] != 1.0:
            s.mul_(self.w1[2])
        step("input_quant")
        per_rank, per_rdma, per_expert, in_rank, _ = self.buffer.get_dispatch_layout(ids, self.experts)
        step("dispatch_layout")
        recv, recv_ids, recv_weights, counts, handle, _ = self.buffer.dispatch(
            (q, s), topk_idx=ids, topk_weights=weights, num_tokens_per_rank=per_rank,
            num_tokens_per_rdma_rank=per_rdma, num_tokens_per_expert=per_expert,
            is_token_in_rank=in_rank, expert_alignment=1, config=self.dispatch_config)
        step("dispatch")
        rx, rs = recv
        total = sum(counts)
        rows = torch.tensor(counts, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        starts = torch.empty_like(rows)
        indices = torch.empty(total, device=x.device, dtype=torch.int32)
        output_index = torch.empty_like(recv_ids)
        grouped = torch.empty((total, self.hidden), device=x.device, dtype=q.dtype)
        scales = torch.empty((total, 1), device=x.device, dtype=torch.float32)
        g1 = torch.empty((total, self.w1[0].size(1)), device=x.device, dtype=torch.bfloat16)
        g2 = torch.empty((total, self.hidden), device=x.device, dtype=torch.bfloat16)
        gathered = torch.empty(rx.shape, device=x.device, dtype=torch.bfloat16)
        step("count_copy_allocate")
        self.scatter(rx, recv_ids.to(torch.int32), rows, starts, grouped, indices, output_index,
                     rs, scales, BLOCK_E=1, is_block_wise=False)
        step("reorder")
        dg.m_grouped_gemm_fp8_fp4_bf16_nt_nopad((grouped, scales), self.w1[:2], g1, indices, m_rows=rows)
        step("gemm1")
        activated = dg.silu_and_mul(dg.ActivationQuantMode.BF16, g1)
        step("activation")
        aq, asc = self.quantize(activated)
        if self.w2[2] != 1.0:
            asc.mul_(self.w2[2])
        step("activation_quant")
        dg.m_grouped_gemm_fp8_fp4_bf16_nt_nopad((aq, asc), self.w2[:2], g2, indices, m_rows=rows)
        step("gemm2")
        self.gather(g2, recv_ids, recv_weights, output_index, gathered)
        step("weighted_gather")
        result, _, _ = self.buffer.combine(gathered, handle, config=self.combine_config)
        step("combine")
        return result


def max_ranks(values):
    t = torch.tensor(values, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def eager_benchmark(functions, warmup, iterations, repeats):
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    results = {name: [] for name in functions}
    for repeat in range(repeats):
        # Alternate A/B ordering. Barriers are outside the timed interval.
        names = list(functions) if repeat % 2 == 0 else list(reversed(functions))
        for name in names:
            dist.barrier()
            torch.cuda.synchronize()
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            begin = time.perf_counter()
            start.record()
            for _ in range(iterations):
                functions[name]()
            end.record()
            end.synchronize()
            wall = (time.perf_counter() - begin) * 1e6 / iterations
            results[name].append(max_ranks([start.elapsed_time(end) * 1000 / iterations, wall]))
    return {name: dict(device_us=statistics.median(x[0] for x in vals),
                       wall_us=statistics.median(x[1] for x in vals), samples=vals)
            for name, vals in results.items()}


def graph_benchmark(fn, iterations, repeats):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        dist.barrier()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(max_ranks([start.elapsed_time(end) * 1000 / iterations])[0])
    return dict(device_us=statistics.median(samples), samples=samples)


def reference(x, ids, weights, w1, w2, checked):
    rank, world = dist.get_rank(), dist.get_world_size()
    q, s = quantize_fp8_rows(x[:checked])
    all_x = gather(q.float() * s)
    all_ids, all_weights = gather(ids[:checked].contiguous()), gather(weights[:checked].contiguous())
    expected = torch.zeros(world * checked, x.size(1), device=x.device)
    for expert in range(w1.size(0)):
        src, tok, slot = torch.where(all_ids == rank * w1.size(0) + expert)
        if not src.numel():
            continue
        projection = (all_x[src, tok] @ w1[expert].float().t()).bfloat16()
        activated = activation(projection).bfloat16()
        q, s = quantize_fp8_rows(activated)
        y = ((q.float() * s) @ w2[expert].float().t()).bfloat16()
        expected.index_add_(0, src * checked + tok, y.float() * all_weights[src, tok, slot, None])
    dist.all_reduce(expected)
    return expected.view(world, checked, x.size(1))[rank].bfloat16()


@torch.inference_mode()
def run(args, group):
    rank, world = dist.get_rank(), dist.get_world_size()
    shape = (7168, 3072, 384, 6) if args.case == "deepseek_v4_pro" else (512, 256, 16, 2)
    h, i, experts, topk = shape
    c = MoECase(args.case, h, i, experts, topk, max(args.tokens), routing="random", precision="w4a8")
    rng = torch.Generator(device="cuda").manual_seed(args.seed + rank)
    raw = [quant_w4a16_mxfp4(torch.randn(experts // world, n, k, dtype=torch.bfloat16,
                                       device="cuda", generator=rng) * amplitude)
           for n, k, amplitude in ((2*i, h, .125), (h, i, .015625))]
    (ref1, w1, s1), (ref2, w2, s2) = raw
    mw1, mw2 = dg.transform_weights_for_mega_moe((w1, s1), (w2, s2), precision="w4a8")
    nw1, nw2 = [dg.preprocess_weights_and_scales(dg.PreprocessMode.W4A8_MXFP4_MMA, False, w, s)
                for _, w, s in raw]
    normal = DeepEPNormal(group, h, nw1, nw2)
    buf = dg.get_symm_buffer_for_mega_moe(group, experts, c.capacity, topk, h,
                                        precision="w4a8", intermediate_hidden=i)
    reports = []
    try:
        for tokens in args.tokens:
            x, ids, weights, _ = make_inputs(c, tokens, rng, edge_cases=True)
            output = torch.empty_like(x)

            def mega(details=None):
                dg.mega_moe_pre_dispatch(x, ids, weights, buf)
                if details:
                    details.step(MEGA_STAGES[0])
                dg.fp8_fp4_mega_moe(output, mw1, mw2, buf)
                if details:
                    details.step(MEGA_STAGES[1])
                return output

            def baseline(details=None):
                return normal.run(x, ids, weights, details)

            # MegaMoE communicates inside the kernel, so precompile it before
            # entering distributed execution. Normal has separate communication.
            compile_kernels(mega)
            torch.cuda.synchronize()
            dist.barrier()
            ym = mega().clone()
            yd = baseline()
            torch.cuda.synchronize()
            checked = min(tokens, args.reference_tokens or tokens)
            ref = reference(x, ids, weights, ref1, ref2, checked)
            errors = max_ranks([float(calc_diff(ym[:checked], ref)), float(calc_diff(yd[:checked], ref)),
                                float(calc_diff(ym, yd)), float((~torch.isfinite(ym)).any()),
                                float((~torch.isfinite(yd)).any())])
            assert max(errors[:3]) < .0015 and max(errors[3:]) == 0, errors
            if rank == 0:
                print("ACCURACY", tokens, errors, flush=True)
            functions = dict(megamoe=mega, deepep_normal=baseline)
            eager = eager_benchmark(functions, args.warmup, args.iterations, args.repeats)
            graph = graph_benchmark(mega, args.iterations, args.repeats)
            stages = {}
            for name, names in (("megamoe", MEGA_STAGES), ("deepep_normal", NORMAL_STAGES)):
                vals = []
                for _ in range(args.repeats):
                    dist.barrier()
                    torch.cuda.synchronize()
                    timer = Events(names)
                    functions[name](timer)
                    torch.cuda.synchronize()
                    vals.append(max_ranks(timer.values()))
                # Component maxima can come from different ranks; do not sum
                # them to claim the end-to-end max-rank latency.
                stages[name] = {key: statistics.median(v[j] for v in vals) for j, key in enumerate(names)}
            report = dict(tokens_per_rank=tokens, shape=shape, ep=world, seed=args.seed,
                          threshold=.0015, checked_tokens_per_rank=checked,
                          reference_diff=dict(megamoe=errors[0], deepep_normal=errors[1]),
                          pairwise_diff=errors[2], eager=eager, megamoe_graph=graph, eager_stage_us=stages)
            reports.append(report)
            if rank == 0:
                print("RESULT", json.dumps(report), flush=True)
                Path(args.output).write_text(json.dumps(dict(args=vars(args), deepgemm=dg.__file__,
                    torch=torch.__version__, deepep=normal.environment, reports=reports), indent=2) + "\n")
            if args.profile:
                for name, fn in functions.items():
                    dist.barrier()
                    torch.cuda.synchronize()
                    context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) if rank == 0 else nullcontext()
                    with context as profiler:
                        dist.barrier()
                        for _ in range(2):
                            with torch.profiler.record_function(name):
                                fn()
                        torch.cuda.synchronize()
                    if rank == 0:
                        path = Path(args.output).with_suffix(f".{tokens}.{name}.trace.json")
                        profiler.export_chrome_trace(str(path))
                        print("TRACE", str(path), flush=True)
                    dist.barrier()
    finally:
        buf.destroy()
        normal.buffer.destroy()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("smoke", "deepseek_v4_pro"), default="deepseek_v4_pro")
    p.add_argument("--tokens", type=int, nargs="+", default=[32, 1024])
    p.add_argument("--seed", type=int, default=8127)
    p.add_argument("--reference-tokens", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", required=True)
    p.add_argument("--profile", action="store_true", help="export rank-0 device traces after timing")
    a = p.parse_args()
    assert min(a.tokens) > 0 and a.iterations > 0 and a.repeats > 0
    with moe_process_group() as group:
        run(a, group)
