r"""PPU MXFP4 MegaMoE public-entry correctness and regression tests.

Single rank:
    python tests/test_mega_moe.py --case smoke_swiglu --num-tokens 33

Distributed correctness:
    torchrun --standalone --nproc-per-node=8 tests/test_mega_moe.py \
        --case deepseek_v4_pro --num-tokens 32 1024 4096 --routing random \
        --reference-tokens 8

Distributed regression suite:
    torchrun --standalone --nproc-per-node=2 -m pytest -q -s -k smoke \
        tests/test_mega_moe.py

Performance and optional DeepEP comparisons live in tests/bench_mega_moe.py.
The legacy --benchmark CLI forwards there.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Direct execution exercises this checkout, even if another is installed.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm
from deep_gemm.testing import calc_diff
from utils import is_ppu1v5_device


# Cases and input/reference helpers are shared with bench_mega_moe.py.
MODEL_TOKENS = (1024, 2048, 4096)
MAX_DIFF = .003


@dataclass(frozen=True)
class MegaMoECase:
    name: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    capacity: int
    tokens: tuple[int, ...] = (1, 33)
    activation: str = "swiglu"
    clamp: float = 0.0
    worlds: tuple[int, ...] = (1, 2, 4, 8)
    reference_tokens: int = 8
    precision: str = "w4a4"
    routing: str = "round_robin"

    def require_world(self):
        world = dist.get_world_size()
        if world not in self.worlds:
            pytest.skip(f"{self.name} requires EP{self.worlds}; launched EP{world}")
        assert self.experts % world == 0


# FP4 dispatch copies scales in 16-byte chunks; smoke H is at least 512.
CASES = (
    MegaMoECase("smoke_swiglu", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8), clamp=2.0),
    MegaMoECase("smoke_step", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8), activation="step", clamp=2.0),
    MegaMoECase("smoke_situ", 512, 256, 8, 2, 129, tokens=(1, 0, 33, 129, 8), activation="situ"),
    MegaMoECase("deepseek_v4_pro", 7168, 3072, 384, 6, 8192, tokens=MODEL_TOKENS, clamp=10.0, worlds=(8,)),
    MegaMoECase("deepseek_v4_flash", 4096, 2048, 256, 6, 4096, tokens=MODEL_TOKENS, clamp=10.0, worlds=(4,)),
    MegaMoECase("glm52", 6144, 2048, 256, 8, 4096, tokens=MODEL_TOKENS, worlds=(8,)),
    MegaMoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=MODEL_TOKENS, activation="situ", worlds=(8, 16)),
    MegaMoECase("qwen38_2_4t", 8192, 2048, 512, 10, 4096, tokens=MODEL_TOKENS, worlds=(8, 16)),
    MegaMoECase("mimo_v26_pro", 6144, 2048, 384, 8, 8192, tokens=(*MODEL_TOKENS, 8192), worlds=(8,)),
)


# Input generation and independent mathematical references.


def _quantize_mxfp4(src_tensor):
    src = src_tensor.float()
    *batch, n, k = src.shape
    grouped = src.reshape(*batch, n, k // 32, 32)
    amax = grouped.abs().amax(-1).clamp_min(1e-10)
    bits = ((amax / 6.0).contiguous().view(torch.int32) + 0x007fffff) & 0x7f800000
    scale = bits.view(torch.float32)
    normalized = (grouped / scale.unsqueeze(-1)).nan_to_num().reshape_as(src)
    v = normalized.abs()
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=src.device)
    code = (v.unsqueeze(-1) > thresholds).sum(-1)
    code += ((v == 0.75) | (v == 1.75) | (v == 3.5)).long()
    code = (code | ((normalized < 0).long() << 3)).to(torch.uint8)
    code = code.reshape(*batch, n, k // 2, 2)
    return (code[..., 0] | (code[..., 1] << 4)).contiguous(), (bits >> 23).to(torch.uint8).contiguous()


def _dequantize_mxfp4(quant_tensor, scale, target_dtype=torch.float32):
    assert quant_tensor.dtype == torch.uint8 and scale.dtype == torch.uint8
    device = quant_tensor.device
    *batch_dims, n, packed_k = quant_tensor.shape
    k = packed_k * 2
    quant_int = quant_tensor.to(torch.int32)
    evens = quant_int & 0xF
    odds = (quant_int >> 4) & 0xF
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lookup_table = torch.tensor(vals + [-v for v in vals], dtype=torch.float32, device=device)
    fp32_tensor = torch.stack([lookup_table[evens], lookup_table[odds]], dim=-1).view(*batch_dims, n, k)
    dq_scale = (scale.to(torch.int32) << 23).view(torch.float32)
    padded_k = dq_scale.shape[-1] * 32
    padded_tensor = F.pad(fp32_tensor, (0, padded_k - k))
    padded_tensor = padded_tensor.view(*batch_dims, n, dq_scale.shape[-1], 32)
    out_padded = (padded_tensor * dq_scale.unsqueeze(-1)).view(*batch_dims, n, padded_k)
    return out_padded[..., :k].to(target_dtype)


def _make_weights(experts, n, k, amplitude, rng):
    # Bound temporary quantization memory for model-sized expert matrices.
    packed, scales = [], []
    for _ in range(experts):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=rng) * amplitude
        q, s = _quantize_mxfp4(weight)
        packed.append(q)
        scales.append(s)
    return torch.stack(packed), torch.stack(scales)


def _make_routes(case, tokens, rng):
    """Distinct top-k experts for balanced, random or 80%-hot routing."""
    token = torch.arange(tokens, device="cuda")[:, None]
    slot = torch.arange(case.topk, device="cuda")[None, :]
    if case.routing == "round_robin":
        return ((token * case.topk + slot + dist.get_rank() * 17) % case.experts).contiguous()
    if case.routing not in ("random", "skewed"):
        raise ValueError(f"unsupported routing distribution: {case.routing}")
    ids = torch.rand(tokens, case.experts, device="cuda", generator=rng).topk(case.topk, dim=1).indices
    if case.routing == "skewed":
        ids = torch.where(token % 5 != 0, slot.expand_as(ids), ids)
    return ids.contiguous()


def _split_expert_weights(weights):
    """Independent allocations expose an ignored or incorrectly offset replica pointer."""
    return tuple(t[:-1].clone() for t in weights), tuple(t[-1:].clone() for t in weights)


def _all_gather(tensor):
    tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors, tensor)
    return torch.stack(tensors)


def _reference_activation(projection, kind="swiglu", clamp=0.0):
    """Round gate/up, their clamps and the activation output to BF16."""
    projection = projection.bfloat16()
    if kind == "situ":
        gate, up = projection.float().chunk(2, -1)
        return (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)
                * 25 * torch.tanh(up / 25)).bfloat16()
    gate, up = projection.chunk(2, -1)
    if clamp > 0:
        up = up.clamp(-clamp, clamp)
        if kind != "step":
            gate = gate.clamp(max=clamp)
    gate = torch.nn.functional.silu(gate.float())
    if kind == "step" and clamp > 0:
        gate = gate.clamp(max=clamp)
    return (gate * up.float()).bfloat16()


def _reference_mega_moe(case, source, ids, weights, l1_weights_raw, l2_weights_raw,
                        num_tokens, checked):
    """Compute the entire routed MoE from source inputs, without kernel intermediates."""
    c = case
    rank_idx, num_ranks = dist.get_rank(), dist.get_world_size()
    num_experts_per_rank = case.experts // num_ranks
    q, scales = _quantize_mxfp4(source[:checked])
    x = _all_gather(_dequantize_mxfp4(q, scales))
    all_ids, all_weights = _all_gather(ids[:checked].contiguous()), _all_gather(weights[:checked].contiguous())
    expected = torch.zeros(num_ranks, checked, c.hidden, device="cuda")
    (l1_weights, l1_weights_sf), (l2_weights, l2_weights_sf) = (l1_weights_raw, l2_weights_raw)
    for expert in range(num_experts_per_rank):
        src, tok, slot = torch.where(all_ids == rank_idx * num_experts_per_rank + expert)
        if not src.numel():
            continue
        projection = x[src, tok] @ _dequantize_mxfp4(l1_weights[expert], l1_weights_sf[expert]).t()
        activated = _reference_activation(projection, c.activation, c.clamp)
        route_weights = all_weights[src, tok, slot, None]
        q, scales = _quantize_mxfp4(activated)
        y = (_dequantize_mxfp4(q, scales)
             @ _dequantize_mxfp4(l2_weights[expert], l2_weights_sf[expert]).t()).bfloat16()
        y = y.float() * route_weights
        expected.view(-1, c.hidden).index_add_(0, src * checked + tok, y)
    dist.all_reduce(expected)
    return expected[rank_idx, :min(num_tokens, checked)].bfloat16()


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-8)).item()


# Distributed lifecycle and public-entry regression checks.


@contextmanager
def _process_group(request=None):
    """Process-group lifecycle for pytest and standalone runs."""
    if not torch.cuda.is_available():
        raise RuntimeError("requires a PPU runtime")
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    if not is_ppu1v5_device():
        message = "MegaMoE requires PPU1.5"
        if request is not None:
            pytest.skip(message)
        raise RuntimeError(message)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    owned = not dist.is_initialized()
    store = None
    if owned:
        kwargs = dict(backend="nccl", timeout=timedelta(seconds=600),
                      device_id=torch.device("cuda", torch.cuda.current_device()))
        if "RANK" not in os.environ:
            store = tempfile.TemporaryDirectory(prefix="deepgemm-moe-test-")
            kwargs.update(init_method=f"file://{store.name}/store", rank=0, world_size=1)
        dist.init_process_group(**kwargs)
    failed = False
    try:
        yield dist.group.WORLD
    except BaseException:
        failed = True
        raise
    finally:
        # A failed rank must exit so torchrun can stop peers waiting in their
        # next collective. Graceful NCCL shutdown can otherwise wait for them.
        distributed_failure = dist.get_world_size() > 1 and (
            failed or (request is not None and request.session.testsfailed))
        if owned and not distributed_failure:
            dist.destroy_process_group()
        if store is not None:
            store.cleanup()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.fixture(scope="session")
def mega_moe_group(request):
    if not torch.cuda.is_available():
        pytest.skip("requires a PPU runtime")
    with _process_group(request) as group:
        yield group


def _compile_kernels(*calls):
    """Compile and load native kernels serially without launching them."""

    previous = deep_gemm.get_compile_mode()
    deep_gemm.set_compile_mode(1)
    try:
        for call in calls:
            call()
    finally:
        deep_gemm.set_compile_mode(previous)


def _check_workspace_options(case, group):
    """All failures must occur before collective symmetric allocation."""

    # Exercise the public constructor's row-index guard without allocating
    # the oversized workspace or requiring a 32-rank process group.
    oversized_group = SimpleNamespace(size=lambda: 32)
    with patch("torch.distributed._symmetric_memory.empty",
               side_effect=AssertionError("overflow must fail before allocation")) as allocate:
        for factory in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises(RuntimeError, match="signed 32-bit row index range"):
                factory(oversized_group, num_experts=1024, num_max_tokens_per_rank=1 << 23,
                        num_topk=16, hidden=512, intermediate_hidden=256)
        allocate.assert_not_called()

    options = dict(num_experts=case.experts, num_max_tokens_per_rank=case.capacity,
                   num_topk=case.topk, hidden=case.hidden, activation=case.activation,
                   intermediate_hidden=case.intermediate)
    invalid = [({"mma_type": "unsupported"}, "requires mma_type"),
               ({"activation": "unsupported"}, "unsupported precision/activation"),
               ({"intermediate_hidden": 127}, "intermediate_hidden"),
               ({"hidden": case.hidden + 1}, "hidden must be divisible"),
               ({"num_topk": 17}, "Invalid MegaMoE"),
               ({"num_max_tokens_per_rank": 0}, "Invalid MegaMoE")]
    for mma_type in ("fp8xfp4", "fp8xfp8", "bf16xbf16"):
        invalid.append(({"mma_type": mma_type}, "requires mma_type"))
    for stage in ("gemm1_config", "gemm2_config"):
        invalid += [({stage: (64, 256)}, "six positive integers"),
                    ({stage: (64, 256, 128, 32, 64, 0)}, "six positive integers"),
                    ({stage: (96, 256, 128, 32, 64, 3)}, "requires BLOCK_M"),
                    ({stage: (64, 384, 128, 32, 64, 3)}, "unsupported BLOCK_N"),
                    ({stage: (64, 256, 48, 32, 64, 3)}, "dividing packed K"),
                    ({stage: (64, 256, 128, 48, 64, 3)}, "invalid warp tile")]
    invalid.append(({"gemm2_config": (256, 256, 128, 64, 64, 64)}, "shared-memory capacity"))
    for legacy in ({"gemm1_block_n": 256}, {"gemm1_block_k": 128}, {"config_override": {}}):
        for factory in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises(TypeError, match=next(iter(legacy))):
                factory(group, **options, **legacy)
    for changed, message in invalid:
        for allocate in (deep_gemm.get_symm_buffer_for_mega_moe, deep_gemm.SymmBuffer):
            with pytest.raises((ValueError, RuntimeError), match=message):
                allocate(group, **{**options, **changed})


def _check_public_entry(entry, y, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options):
    """Reject malformed calls before advancing the collective compute protocol."""

    deep_gemm.mega_moe_pre_dispatch(source, ids, weights, sym_buffer)
    with pytest.raises((ValueError, RuntimeError), match="unsupported precision/activation"):
        entry(y, l1_weights, l2_weights, sym_buffer, **{**options, "activation": "unsupported"})
    unsupported_activation = "swiglu_oai"
    with pytest.raises((ValueError, RuntimeError), match="unsupported precision/activation"):
        entry(y, l1_weights, l2_weights, sym_buffer, **{**options, "activation": unsupported_activation})
    for changed, message in (
        ({"recipe": (128, 128, 128)}, "recipe=\\(1, 1, 32\\)"),
        ({"recipe": (1, 1, 16)}, "recipe=\\(1, 1, 32\\)"),
        ({"activation_clamp": float("nan")}, "activation_clamp"),
        ({"situ_beta": float("nan")}, "SiTU parameters|situ_beta"),
        ({"situ_linear_beta": 0.0}, "SiTU parameters|situ_beta"),
    ):
        with pytest.raises(RuntimeError, match=message):
            entry(y, l1_weights, l2_weights, sym_buffer, **{**options, **changed})
    num_experts_per_rank = sym_buffer.num_experts // sym_buffer.num_ranks
    for stats in (torch.zeros(num_experts_per_rank, device=y.device),
                  torch.zeros(num_experts_per_rank, dtype=torch.int32),
                  torch.zeros(2 * num_experts_per_rank, device=y.device, dtype=torch.int32)[::2]):
        with pytest.raises(RuntimeError, match="tensor device, dtype or layout"):
            entry(y, l1_weights, l2_weights, sym_buffer,
                  **{**options, "cumulative_local_expert_recv_stats": stats})
    with pytest.raises(RuntimeError, match="one INT32 element per local physical expert"):
        entry(y, l1_weights, l2_weights, sym_buffer,
              **{**options, "cumulative_local_expert_recv_stats": torch.zeros(
                  num_experts_per_rank + 1, device=y.device, dtype=torch.int32)})
    with pytest.raises(RuntimeError, match="tensor device, dtype or layout"):
        entry(y.float(), l1_weights, l2_weights, sym_buffer, **options)
    bad_w2 = (l2_weights[0][:, :-1].contiguous(), l2_weights[1])
    with pytest.raises(RuntimeError, match="tensor shape"):
        entry(y, l1_weights, bad_w2, sym_buffer, **options)
    # A caller may fill public views directly and choose tokens through y.
    # Preserve exact staged bytes so this checks the API contract independently
    # of Torch versus kernel quantization at FP4 rounding boundaries.
    count = source.size(0)
    views = (sym_buffer.x, sym_buffer.x_sf, sym_buffer.topk_idx, sym_buffer.topk_weights)
    staged = [view[:count].clone() for view in views]
    entry(y, l1_weights, l2_weights, sym_buffer, **options)
    expected = y.clone()
    # The common argument order matches SM90; PPU extensions remain keyword-only.
    entry(y, l1_weights, l2_weights, sym_buffer, None, (1, 1, 32),
          options["activation"], options["activation_clamp"], options["fast_math"])
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    if options.get("fast_math", True):
        default_options = {name: value for name, value in options.items() if name != "fast_math"}
        entry(y, l1_weights, l2_weights, sym_buffer, **default_options)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
    for view, value in zip(views, staged):
        view[:count].copy_(value)
    entry(y[:0], l1_weights, l2_weights, sym_buffer, **options)
    entry(y, l1_weights, l2_weights, sym_buffer, **options)
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    _check_offset_views(entry, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options)


def _check_recv_stats(pipeline, y, routing_ids, routing_weights, num_experts):
    """Cumulative receive statistics survive reuse and change with replayed routes."""
    rank, world = dist.get_rank(), dist.get_world_size()
    local_experts = num_experts // world
    stats = torch.full((local_experts,), 7, dtype=torch.int32, device=y.device)
    expected_stats = stats.clone()
    saved_ids, saved_weights = routing_ids.clone(), routing_weights.clone()

    def received():
        ids = routing_ids.flatten()
        ids = ids[(ids >= 0) & (ids < num_experts)]
        counts = torch.zeros(num_experts, dtype=torch.int32, device=y.device)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        dist.all_reduce(counts)
        return counts[rank * local_experts:(rank + 1) * local_experts]

    try:
        _compile_kernels(lambda: pipeline(stats))
        torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
        baseline = pipeline().clone()
        for _ in range(3):
            pipeline(stats)
            expected_stats += received()
            torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
            torch.testing.assert_close(y, baseline, rtol=0, atol=0)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pipeline(stats)
        # Capture records work; only actual launches may increment the counters.
        torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
        hot = (torch.arange(routing_ids.size(1), device=y.device) +
               ((rank + 1) % world) * local_experts) % num_experts
        for ids, zero_weights in ((hot.expand_as(routing_ids), False),
                                  (saved_ids, True),
                                  (torch.full_like(saved_ids, -1), False),
                                  (saved_ids, False)):
            routing_ids.copy_(ids)
            routing_weights.copy_(0 * saved_weights if zero_weights else saved_weights)
            baseline = pipeline().clone()
            count = received()
            for _ in range(2):
                graph.replay()
                expected_stats += count
                torch.testing.assert_close(stats, expected_stats, rtol=0, atol=0)
                torch.testing.assert_close(y, baseline, rtol=0, atol=0)
    finally:
        routing_ids.copy_(saved_ids)
        routing_weights.copy_(saved_weights)


def _check_offset_views(entry, l1_weights, l2_weights, sym_buffer, source, ids, weights, **options):
    """Offset views must match aligned complete calls without touching their guards."""

    def forward(input_tensor, y):
        deep_gemm.mega_moe_pre_dispatch(input_tensor, ids, weights, sym_buffer)
        entry(y, l1_weights, l2_weights, sym_buffer, **options)
        return y

    aligned_input = source.clone()
    aligned_output = torch.empty_like(source)
    _compile_kernels(lambda: forward(aligned_input, aligned_output))
    torch.cuda.synchronize()
    dist.barrier()
    expected = forward(aligned_input, aligned_output).clone()

    def make_view(offset):
        storage = torch.full((source.numel() + offset + 8,), 13.0,
                             dtype=source.dtype, device=source.device)
        view = storage[offset:offset + source.numel()].view_as(source)
        assert view.is_contiguous() and view.storage_offset() == offset
        return storage, view

    def check_guards(storage, offset):
        assert torch.all(storage[:offset] == 13)
        assert torch.all(storage[offset + source.numel():] == 13)

    # Exercise input-only, output-only and joint offsets, including every BF16
    # alignment modulo 16 bytes. The empty peer still participates in compute.
    for input_offset, output_offset in ((1, 0), (0, 1), *((i, i) for i in range(1, 8))):
        input_storage, offset_input = make_view(input_offset)
        output_storage, offset_output = make_view(output_offset)
        offset_input.copy_(source)
        actual = forward(offset_input, offset_output)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(offset_input, source, rtol=0, atol=0)
        check_guards(input_storage, input_offset)
        check_guards(output_storage, output_offset)

    # Replay from offset allocations too; changed inputs and routes must not
    # consume data left by an eager call or by an earlier workspace generation.
    _check_graph_replay(lambda: forward(offset_input, offset_output), expected,
                       input_tensor=offset_input, routing_ids=ids, routing_weights=weights,
                       combine=sym_buffer._combine, num_experts=sym_buffer.num_experts)
    check_guards(input_storage, input_offset)
    check_guards(output_storage, output_offset)


def _check_graph_replay(pipeline, expected, *, input_tensor=None, routing_weights=None, routing_ids=None,
                       check_reference=None, combine=None, num_experts=None, barrier_flag=None,
                       dispatch_counts=None):
    """Recycle state with changing data; check eager and replay against a Torch oracle."""
    if barrier_flag is not None:
        # Fault injection at an idle point: exercise the signed generation
        # wraparound without billions of forwards, including on empty ranks.
        torch.cuda.synchronize()
        dist.barrier()
        barrier_flag.fill_(0x7ffffffe)
        torch.cuda.synchronize()
        dist.barrier()
    for _ in range(3):
        pipeline()
    torch.cuda.synchronize()
    dist.barrier()
    if barrier_flag is not None:
        assert barrier_flag.item() < 0, "combine barrier generation did not wrap"
    if dispatch_counts is not None:
        assert torch.count_nonzero(dispatch_counts) == 0, "dispatch counts were not recycled"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = pipeline()

    rank, world = dist.get_rank(), dist.get_world_size()
    check_peers = num_experts is not None and world > 1
    generation = 0

    def compare(expect_zero=False):
        nonlocal generation
        if combine is not None:
            combine.fill_(float("nan"))
            result.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
        # Stagger a rotating rank after poisoning all receive buffers. The
        # complete entry must synchronize its own scatter and combine stages.
        if check_peers and rank == generation % world:
            time.sleep(.003)
        # Replay must produce its own intermediates before eager can refresh them.
        graph.replay()
        replayed = result.clone()
        torch.cuda.synchronize()
        dist.barrier()
        if dispatch_counts is not None:
            assert torch.count_nonzero(dispatch_counts) == 0, "graph replay did not recycle dispatch counts"
        if combine is not None:
            combine.fill_(float("nan"))
            result.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
        if check_peers and rank == generation % world:
            time.sleep(.003)
        eager = pipeline().clone()
        torch.cuda.synchronize()
        if dispatch_counts is not None:
            assert torch.count_nonzero(dispatch_counts) == 0, "eager forward did not recycle dispatch counts"
        assert replayed.isfinite().all(), "recycled routes read an unproduced slot"
        torch.testing.assert_close(replayed, eager, rtol=0, atol=0)
        if expect_zero:
            assert torch.count_nonzero(replayed) == 0, "inactive routes must produce exact zeros"
        if check_reference is not None:
            check_reference(replayed)
        dist.barrier()
        generation += 1

    if input_tensor is not None:
        saved = input_tensor.clone()
        for factor in (0.0, -1.0, 1.0):
            input_tensor.copy_(saved * factor)
            compare()
    else:
        compare()
    if routing_ids is not None:
        saved_ids = routing_ids.clone()
        hot_routes = torch.arange(saved_ids.shape[1], device=saved_ids.device).expand_as(saved_ids)
        if check_peers:
            experts_per_rank = num_experts // world
            # Send to the next rank, then concentrate work on the last rank.
            # The last rank has no input in the one-token edge case but must
            # still compute remote experts. Wrap slots for small expert counts.
            for destination in ((rank + 1) % world, world - 1):
                routing_ids.copy_((destination * experts_per_rank + hot_routes) % num_experts)
                compare()
            rng = torch.Generator(device=saved_ids.device).manual_seed(20261022 + rank)
            routing_ids.copy_(torch.rand(saved_ids.shape[0], num_experts,
                                        device=saved_ids.device, generator=rng)
                              .topk(saved_ids.shape[1], dim=1).indices)
            compare()
        for changed, expect_zero in ((hot_routes, False), (torch.full_like(saved_ids, -1), True),
                                     (saved_ids, False)):
            routing_ids.copy_(changed)
            compare(expect_zero=expect_zero)
        # Return to original routing before changing weights.
    if routing_weights is not None:
        saved_weights = routing_weights.clone()
        slots = torch.arange(saved_weights.shape[1], device=saved_weights.device)
        # Include tiny, nonuniform and zero routing weights. Do not renormalize:
        # callers may supply weights whose sum is not one.
        factors = torch.where(slots == 0, 0.0, 10.0 ** (-slots.float()))
        for changed, expect_zero in ((saved_weights * factors, False),
                                     (torch.zeros_like(saved_weights), True), (saved_weights, False)):
            routing_weights.copy_(changed)
            compare(expect_zero=expect_zero)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


# Complete-call numerical and state-reuse checks.


def _check_mega_moe(case, source, requested, pipeline, reference, *, combine, check=True,
                reference_tokens=0, max_diff=MAX_DIFF, max_relative_l2=None,
                replicas=False, edge_cases=False, verbose=True,
                routing_weights=None, routing_ids=None, workspace=None):
    """Check a complete forward, then reuse its workspace with graph replay."""
    rank, world, count = dist.get_rank(), dist.get_world_size(), source.shape[0]
    _compile_kernels(pipeline)
    block_info = workspace._gemm2_block_info if workspace is not None else None
    if block_info is not None:
        # Scheduler metadata must be regenerated, including after route changes
        # and replay with empty experts; old contents must never be consumed.
        block_info.fill_(-1)
    combine.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier()
    actual = pipeline().clone()
    counts = _all_gather(torch.tensor(count, device="cuda")).tolist()
    report = dict(case=case.name, precision=case.precision, ep=world, tokens_per_rank=counts,
                  hidden=case.hidden, intermediate=case.intermediate, experts=case.experts,
                  capacity=case.capacity, topk=case.topk, activation=case.activation,
                  scatter_dtype="bf16", routing=case.routing, split_weights=replicas,
                  edge_cases=edge_cases, correctness=dict(status="not_checked"))
    if check:
        checked = min(requested, reference_tokens) if reference_tokens else requested
        replay_metrics = []
        def check_reference(result):
            expected = reference(checked)
            assert result.isfinite().all(), (rank, "nonfinite pipeline output")
            sample = result[:min(count, checked)]
            metrics = torch.tensor([float(calc_diff(sample, expected)), _relative_l2(sample, expected)],
                                   device="cuda")
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            assert metrics[0].item() < max_diff, (case.name, "pipeline diff", metrics.tolist())
            if max_relative_l2 is not None:
                assert metrics[1].item() < max_relative_l2, (case.name, "pipeline relative L2", metrics.tolist())
            replay_metrics.append(metrics.tolist())
            return metrics

        metrics = check_reference(actual)
        report["correctness"] = dict(status="passed", max_pipeline_diff=metrics[0].item(),
                                     max_relative_l2=metrics[1].item(),
                                     checked_tokens_per_rank=[min(n, checked) for n in counts])
        _check_graph_replay(pipeline, actual, input_tensor=source, routing_weights=routing_weights,
                           routing_ids=routing_ids, check_reference=check_reference, combine=combine,
                           num_experts=case.experts if edge_cases else None,
                           barrier_flag=workspace._barrier_flag if edge_cases and workspace is not None else None,
                           dispatch_counts=workspace._dispatch_counts if edge_cases and workspace is not None else None)
        report["correctness"]["max_replay_pipeline_diff"] = max(m[0] for m in replay_metrics[1:])
        report["correctness"]["max_replay_relative_l2"] = max(m[1] for m in replay_metrics[1:])
        report["correctness"]["changed_routes_reference_checked"] = True
        if edge_cases and world > 1:
            report["correctness"]["cross_rank_visibility_checked"] = True
        if routing_weights is not None:
            report["correctness"]["routing_weight_replay_checked"] = True
        if workspace is not None:
            assert workspace._gemm2_block_info is block_info, "GEMM2 replaced workspace scratch"
            report["correctness"]["workspace_reuse_checked"] = True
    if verbose and rank == 0:
        print(json.dumps(report), flush=True)
    dist.barrier()
    return report


# End-to-end correctness driver: setup -> public complete call -> reference.


@torch.inference_mode()
def _run_mega_moe(case, group, *, check=True, reference_tokens=0, edge_cases=False,
                  splits=(False,), verbose=True, workspace_options=None, fast_math=True):
    """Run public MegaMoE calls against independent and stateful references."""
    rank_idx, num_ranks = dist.get_rank(), dist.get_world_size()
    num_experts_per_rank = case.experts // num_ranks
    if check and case.name.startswith("smoke"):
        _check_workspace_options(case, group)
    rng = torch.Generator(device="cuda").manual_seed(20261016 + rank_idx)
    x_bf16 = torch.randn(case.capacity, case.hidden, device="cuda", dtype=torch.bfloat16, generator=rng) * .5
    l1_weights_raw = _make_weights(num_experts_per_rank, 2 * case.intermediate, case.hidden, .5, rng)
    l2_weights_raw = _make_weights(num_experts_per_rank, case.hidden, case.intermediate, .125, rng)
    l1_weights, l2_weights = deep_gemm.transform_weights_for_mega_moe(
        l1_weights_raw, l2_weights_raw, activation=case.activation)
    sym_buffer = deep_gemm.SymmBuffer(
        group, case.experts, case.capacity, case.topk, case.hidden, case.intermediate,
        activation=case.activation, **(workspace_options or {}))
    # Private storage is used only for NaN and workspace-reuse fault injection.
    combine = sym_buffer._combine
    split_l1_weights = split_l2_weights = None
    reports = []
    try:
        for token_index, requested in enumerate(case.tokens):
            empty_rank = edge_cases and num_ranks > 1 and requested == 1 and rank_idx == num_ranks - 1
            num_tokens = 0 if empty_rank else requested
            topk_idx = _make_routes(case, case.capacity, rng)
            topk_weights = torch.rand(case.capacity, case.topk, device="cuda", generator=rng).softmax(-1)
            topk_idx[num_tokens:], topk_weights[num_tokens:] = -1, 0
            if edge_cases and num_tokens > 1:
                # Invalid IDs must mask even a nonzero router weight.
                topk_idx[0, 0] = -1
                topk_weights[1, 0] = 0
            for split in splits:
                x = x_bf16[:num_tokens]
                routed_idx, routed_weights = topk_idx[:num_tokens], topk_weights[:num_tokens]
                y = torch.empty_like(x)
                if split and split_l1_weights is None:
                    split_l1_weights = _split_expert_weights(l1_weights)
                    split_l2_weights = _split_expert_weights(l2_weights)
                l1_master, l1_replicas = split_l1_weights if split else (l1_weights, None)
                l2_master, l2_replicas = split_l2_weights if split else (l2_weights, None)

                # Exercise the complete public call, including BF16 input preparation.
                def run_fused(cumulative_local_expert_recv_stats=None):
                    deep_gemm.mega_moe_pre_dispatch(x, routed_idx, routed_weights, sym_buffer)
                    deep_gemm.fp4_fp4_mega_moe(
                        y, l1_master, l2_master, sym_buffer,
                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                        activation=case.activation, activation_clamp=case.clamp or None,
                        fast_math=fast_math,
                        l1_weights_replicas=l1_replicas, l2_weights_replicas=l2_replicas)
                    return y

                if check and case.name.startswith("smoke") and not split and token_index == 0:
                    _check_public_entry(
                        deep_gemm.fp4_fp4_mega_moe, y, l1_master, l2_master, sym_buffer,
                        x, routed_idx, routed_weights, activation=case.activation,
                        activation_clamp=case.clamp or None, fast_math=fast_math)
                if check and case.name.startswith("smoke") and requested in (1, 33):
                    _check_recv_stats(run_fused, y, routed_idx, routed_weights, case.experts)
                report = _check_mega_moe(
                    case, x, requested, run_fused,
                    lambda checked: _reference_mega_moe(
                        case, x_bf16, topk_idx, topk_weights, l1_weights_raw, l2_weights_raw,
                        num_tokens, checked),
                    combine=combine, workspace=sym_buffer, check=check, reference_tokens=reference_tokens,
                    replicas=split, edge_cases=edge_cases, verbose=False,
                    routing_weights=routed_weights, routing_ids=routed_idx)
                if check and workspace_options:
                    meta = sym_buffer.get_gemm2_metadata()
                    for option, rows in (("gemm1_config", meta.src_m_rows),
                                         ("gemm2_config", meta.m_rows)):
                        if option in workspace_options:
                            bm = workspace_options[option][0]
                            expected_rows = (meta.valid_m_rows + bm - 1) // bm * bm
                            torch.testing.assert_close(rows, expected_rows, rtol=0, atol=0)
                report["fast_math"] = fast_math
                if verbose and rank_idx == 0:
                    print(json.dumps(report), flush=True)
                reports.append(report)
    finally:
        sym_buffer.destroy()
    return reports


@pytest.mark.parametrize("case,fast_math", [
    pytest.param(case, fast_math, id=case.name + ("" if fast_math else "_precise"))
    for case in CASES
    for fast_math in ((True, False) if case.name.startswith("smoke") else (True,))
])
def test_mega_moe(case, fast_math, mega_moe_group):
    case.require_world()
    splits = (False, True) if case.experts // dist.get_world_size() > 1 else (False,)
    _run_mega_moe(case, mega_moe_group, edge_cases=True, splits=splits, verbose=False,
                  reference_tokens=0 if case.name.startswith("smoke") else case.reference_tokens,
                  fast_math=fast_math)


@pytest.mark.parametrize("options", [
    {"gemm1_config": (64, 256, 128, 32, 64, 3)},
    {"gemm1_config": (64, 128, 128, 32, 64, 3)},
    {"gemm2_config": (64, 256, 128, 32, 64, 3)},
    {"gemm1_config": (128, 256, 64, 64, 64, 3),
     "gemm2_config": (64, 256, 128, 32, 64, 3)},
    {"gemm1_config": (64, 256, 128, 32, 64, 3),
     "gemm2_config": (128, 256, 64, 64, 64, 3)},
], ids=["smoke_g1_only", "smoke_g1_bn128", "smoke_g2_only", "smoke_g1_larger", "smoke_g2_larger"])
def test_mega_moe_tiles(options, mega_moe_group):
    # Reuse the public-entry reference, changing routes, empty-rank and graph
    # checks with different source and GEMM2 padding in the same workspace.
    case = CASES[0]
    case.require_world()
    splits = (False, True) if case.experts // dist.get_world_size() > 1 else (False,)
    _run_mega_moe(case, mega_moe_group, edge_cases=True, splits=splits, verbose=False,
                  workspace_options=options)


# Standalone torchrun entry; pytest uses the same driver and case table.


def main():
    if "--benchmark" in sys.argv[1:]:
        from bench_mega_moe import main as benchmark_main
        benchmark_main([arg for arg in sys.argv[1:] if arg != "--benchmark"])
        return
    cases = CASES
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=[c.name for c in cases], default=cases[0].name)
    parser.add_argument("--num-tokens", "--tokens", dest="tokens", type=int, nargs="+",
                        help="tokens per rank; defaults to the case table")
    parser.add_argument("--num-max-tokens-per-rank", "--capacity", dest="capacity", type=int,
                        help="workspace token capacity")
    parser.add_argument("--routing", choices=("round_robin", "random", "skewed"), default="round_robin")
    parser.add_argument("--fast-math", type=int, choices=(0, 1), default=1,
                        help="approximate GEMM1 activation math (default: 1); 0 uses expf/division/tanhf")
    parser.add_argument("--reference-tokens", type=int, default=0, help="check first N tokens/rank; 0 checks all")
    parser.add_argument("--replicas", action="store_true", help="use separate master/replica weight buffers")
    parser.add_argument("--edge-cases", action="store_true",
                        help="include empty ranks, masked routes and staggered cross-rank replay")
    args = parser.parse_args()
    case = next(c for c in cases if c.name == args.case)
    case = replace(case, tokens=tuple(args.tokens) if args.tokens is not None else case.tokens,
                   capacity=args.capacity if args.capacity is not None else case.capacity, routing=args.routing)
    if case.capacity < 1 or any(n < 0 or n > case.capacity for n in case.tokens):
        parser.error("tokens must be nonnegative and no larger than a positive capacity")
    if args.reference_tokens < 0:
        parser.error("reference-tokens must be nonnegative")
    with _process_group() as group:
        world = dist.get_world_size()
        if world not in case.worlds or case.experts % world:
            raise ValueError(f"{case.name} requires EP{case.worlds}; launched EP{world}")
        if args.replicas and case.experts // world < 2:
            raise ValueError("--replicas requires at least two local experts")
        _run_mega_moe(case, group, reference_tokens=args.reference_tokens,
                      edge_cases=args.edge_cases, splits=(args.replicas,),
                      fast_math=bool(args.fast_math))


if __name__ == "__main__":
    main()
