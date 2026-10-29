"""W4A8 MegaMoE through public APIs; run with torchrun or pytest on ZW-M890P.

    torchrun --standalone --nproc-per-node=2 tests/test_moe_w4a8.py --case smoke --tokens 1 33
"""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deep_gemm as dg
from moe_test_utils import (MoECase, activation, gather, make_inputs, moe_group,
                           quant_w4a16_mxfp4, quantize_fp8_rows, run_cli, run_forward)

CASES = (
    MoECase("smoke", 512, 256, 16, 2, 129, tokens=(1, 0, 33, 129, 8, 1), precision="w4a8"),
    MoECase("smoke_situ", 512, 512, 32, 2, 129, tokens=(1, 0, 33, 129, 8, 1),
            activation="situ", precision="w4a8", worlds=(1, 2, 4, 8, 16, 32)),
    MoECase("smoke_cached_quant", 512, 512, 16, 2, 33, tokens=(1, 0, 33), precision="w4a8"),
    MoECase("tail", 256, 384, 16, 2, 65, tokens=(17, 65), precision="w4a8"),
    MoECase("experts128", 256, 128, 256, 2, 33, tokens=(1, 33), worlds=(2,), precision="w4a8"),
    # More than one readiness window, with a capacity much larger than the
    # active input. Reuse counts across empty, sparse and changing routes.
    MoECase("workspace_reuse", 512, 256, 128, 2, 4096, tokens=(65, 0, 1, 33, 129, 8, 65),
            worlds=(2,), precision="w4a8"),
    MoECase("deepseek_v4_pro", 7168, 3072, 384, 6, 4096, tokens=(32, 1024, 4096),
            worlds=(8,), precision="w4a8", reference_tokens=8),
    MoECase("kimi_k3", 3584, 3072, 896, 16, 4096, tokens=(4, 8, 16, 32, 64, 128, 1024, 2048, 4096),
            activation="situ", worlds=(8, 16, 32), precision="w4a8", reference_tokens=8),
)


def split_weights(w):
    weight, scale, inv = w
    at = max(1, weight.size(0) // 2)
    return (weight[:at], scale[:at], inv), (weight[at:], scale[at:], inv)


@torch.inference_mode()
def run_case(case, group, *, check=True, reference_tokens=0, edge_cases=False,
             splits=(False,), benchmark=None, verbose=True, block_m=None, shifted=False, independent_replicas=False):
    c = case
    rank, world = dist.get_rank(), dist.get_world_size()
    epr = c.experts // world
    rng = torch.Generator(device="cuda").manual_seed(8127 + rank)
    weights = []
    for n, k, amplitude in ((2*c.intermediate, c.hidden, .125), (c.hidden, c.intermediate, .015625)):
        values = torch.randn(epr, n, k, dtype=torch.bfloat16, device="cuda", generator=rng) * amplitude
        # Force load-time exponent rebasing, while retaining ordinary output magnitudes.
        if shifted:
            values *= 2.0 ** (-16 if k == c.hidden else 32)
        if independent_replicas:
            values[epr // 2:] *= 2.0 ** (-16 if k == c.hidden else 32)
        weights.append(quant_w4a16_mxfp4(values))
    (ref1, w1, s1), (ref2, w2, s2) = weights
    if independent_replicas:
        at = epr // 2
        packed1, packed2 = dg.transform_weights_for_mega_moe((w1[:at], s1[:at]), (w2[:at], s2[:at]), precision="w4a8")
        replica1, replica2 = dg.transform_weights_for_mega_moe((w1[at:], s1[at:]), (w2[at:], s2[at:]), precision="w4a8")
        assert packed1[2] != replica1[2] and packed2[2] != replica2[2]
    else:
        packed1, packed2 = dg.transform_weights_for_mega_moe((w1, s1), (w2, s2), precision="w4a8")
    if shifted:
        assert packed1[2] != 1.0 and packed2[2] != 1.0
    buf = dg.get_symm_buffer_for_mega_moe(group, c.experts, c.capacity, c.topk, c.hidden,
        precision="w4a8", intermediate_hidden=c.intermediate, block_m=block_m)
    reports = []
    try:
        for split in splits:
            if independent_replicas:
                assert split
                g1, r1, g2, r2 = packed1, replica1, packed2, replica2
            else:
                g1, r1 = split_weights(packed1) if split else (packed1, None)
                g2, r2 = split_weights(packed2) if split else (packed2, None)
            for requested in c.tokens:
                source, ids, weights, count = make_inputs(c, requested, rng, edge_cases)
                output = torch.empty(count, c.hidden, device="cuda", dtype=torch.bfloat16)

                def forward():
                    dg.mega_moe_pre_dispatch(source[:count], ids[:count], weights[:count], buf)
                    dg.fp8_fp4_mega_moe(output, g1, g2, buf, activation=c.activation,
                        activation_limit=c.clamp or None, w1_replicas=r1, w2_replicas=r2)
                    return output

                def reference(checked):
                    q, s = quantize_fp8_rows(source[:checked])
                    x = gather(q.float() * s)
                    all_ids, all_weights = gather(ids[:checked].contiguous()), gather(weights[:checked].contiguous())
                    expected = torch.zeros(world * checked, c.hidden, device="cuda")
                    for expert in range(epr):
                        src, tok, slot = torch.where(all_ids == rank * epr + expert)
                        if not src.numel():
                            continue
                        projection = (x[src, tok] @ ref1[expert].float().t()).bfloat16()
                        a = activation(projection, c.activation, c.clamp).bfloat16()
                        q, s = quantize_fp8_rows(a)
                        y = ((q.float() * s) @ ref2[expert].float().t()).bfloat16()
                        expected.index_add_(0, src * checked + tok, y.float() * all_weights[src, tok, slot, None])
                    dist.all_reduce(expected)
                    return expected.view(world, checked, c.hidden)[rank, :min(count, checked)].bfloat16()

                reports.append(run_forward(c, source[:count], requested, forward, reference,
                    combine=buf._combine, check=check, reference_tokens=reference_tokens, max_diff=.0015,
                    replicas=split, edge_cases=edge_cases, benchmark=benchmark, verbose=verbose,
                    routing_weights=weights[:count], routing_ids=ids[:count], workspace=buf))
    finally:
        buf.destroy()
    return reports


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_pipeline(case, moe_group):
    case.require_world()
    splits = (False, True) if case.experts // moe_group.size() > 1 else (False,)
    run_case(case, moe_group, edge_cases=True, splits=splits, verbose=False,
             reference_tokens=case.reference_tokens if case.name in ("kimi_k3", "deepseek_v4_pro") else 0)


@pytest.mark.parametrize("block_m", (16, 32, 64, 128))
def test_tiles_and_activation(block_m, moe_group):
    case = replace(CASES[0], tokens=(33,), activation="swiglu_oai", clamp=2.0, routing="skewed")
    run_case(case, moe_group, block_m=block_m, edge_cases=True, verbose=False)


@pytest.mark.parametrize("block_m", (None, 32))
def test_dynamic_tiles(block_m, moe_group):
    case = next(c for c in CASES if c.name == "kimi_k3")
    case.require_world()
    # One capacity must support changing tile/padding geometry, an explicit
    # G1 override, empty inputs and graph replay without caller-side resets.
    case = replace(case, tokens=(4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 0, 4), routing="random")
    run_case(case, moe_group, block_m=block_m, edge_cases=True, verbose=False,
             reference_tokens=8)


@pytest.mark.parametrize("hidden", (256, 512, 768))
def test_scatter_widths(hidden, moe_group):
    # Exercise bulk-store row widths with partial M tiles and changing routes.
    case = replace(CASES[0], hidden=hidden, intermediate=512,
                   tokens=(1, 33, 0, 65), routing="skewed")
    run_case(case, moe_group, edge_cases=True, splits=(False, True), verbose=False)


@pytest.mark.parametrize("routing", ("random", "skewed"))
def test_quantize_lookup(routing, moe_group):
    # Change the valid-row count across both expert-search regimes, including
    # empty experts and workspace reuse after a large batch.
    case = replace(CASES[0], hidden=256, intermediate=128, experts=128,
                   capacity=4096, tokens=(33, 4096, 0, 1, 33),
                   worlds=(2,), routing=routing)
    case.require_world()
    run_case(case, moe_group, edge_cases=True, verbose=False)


def test_weight_scale_shift(moe_group):
    run_case(replace(CASES[0], tokens=(17,)), moe_group, shifted=True, edge_cases=True,
             splits=(False, True), verbose=False)


def test_replica_scale_shift(moe_group):
    run_case(replace(CASES[0], tokens=(17,), routing="random"), moe_group,
             independent_replicas=True, splits=(True,), edge_cases=True, verbose=False)


@pytest.mark.parametrize("ranks", (16, 32, 33, 64))
def test_rank_config(ranks, moe_group):
    from deep_gemm import deep_gemm_cpp

    options = dict(num_ranks=ranks, num_experts=2 * ranks, num_topk=2, hidden=512,
                   num_max_tokens_per_rank=33, precision="w4a8", intermediate_hidden=256)
    if ranks <= 32:
        config = deep_gemm_cpp._mega_moe_dispatch_config(**options)
        assert config["NUM_RANKS"] == ranks
        assert config["NUM_EXPERTS_PER_RANK"] == 2
    else:
        with pytest.raises(RuntimeError, match="Invalid MegaMoE ranks"):
            deep_gemm_cpp._mega_moe_dispatch_config(**options)


def test_invalid_options(moe_group):
    options = dict(num_experts=max(16, 2 * moe_group.size()), num_max_tokens_per_rank=33, num_topk=2, hidden=512,
                   precision="w4a8", intermediate_hidden=256)
    for change, message in (({"block_m": 256}, "W4A8 block_m"),
                            ({"hidden": 384}, "hidden must be divisible"),
                            ({"num_experts": 129 * moe_group.size()}, "at most 128 local experts"),
                            ({"intermediate_hidden": 64}, "intermediate_hidden")):
        with pytest.raises(RuntimeError, match=message):
            dg.get_symm_buffer_for_mega_moe(moe_group, **{**options, **change})


def test_invalid_forward(moe_group):
    experts = max(16, 2 * moe_group.size())
    e = experts // moe_group.size()
    w1 = torch.zeros(e, 512, 256, dtype=torch.uint8, device="cuda")
    w2 = torch.zeros(e, 512, 128, dtype=torch.uint8, device="cuda")
    s1 = torch.full((e, 512, 16), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((e, 512, 8), 127, dtype=torch.uint8, device="cuda")
    g1, g2 = dg.transform_weights_for_mega_moe((w1, s1), (w2, s2), precision="w4a8")
    buf = dg.get_symm_buffer_for_mega_moe(moe_group, experts, 1, 2, 512,
        precision="w4a8", intermediate_hidden=256)
    y = torch.empty(1, 512, dtype=torch.bfloat16, device="cuda")
    try:
        with pytest.raises(RuntimeError, match="call mega_moe_pre_dispatch"):
            dg.fp8_fp4_mega_moe(y, g1, g2, buf)
        dg.mega_moe_pre_dispatch(torch.zeros_like(y),
            torch.zeros(1, 2, dtype=torch.int64, device="cuda"),
            torch.ones(1, 2, device="cuda"), buf)
        for opts, message in (({"activation": "step"}, "W4A8 supports"),
                              ({"situ_beta": 2.0}, "SiTU parameters"),
                              ({"activation": "situ", "activation_limit": 2.0}, "activation_limit"),
                              ({"activation": "situ", "situ_beta": float("nan")}, "situ_beta"),
                              ({"activation": "situ", "situ_linear_beta": 0.0}, "situ_beta"),
                              ({"activation_limit": float("nan")}, "activation_limit"),
                              ({"activation_alpha": 0.0}, "activation_alpha"),
                              ({"w1_replicas": g1}, "same replica partition")):
            with pytest.raises(RuntimeError, match=message):
                dg.fp8_fp4_mega_moe(y, g1, g2, buf, **opts)
        with pytest.raises(RuntimeError, match="weight_inv_scale"):
            dg.fp8_fp4_mega_moe(y, (*g1[:2], 0.0), g2, buf)
        with pytest.raises(RuntimeError, match="tensor shape"):
            dg.fp8_fp4_mega_moe(y, g1, (g2[0], s2, g2[2]), buf)
    finally:
        buf.destroy()


if __name__ == "__main__":
    run_cli(CASES, run_case, __doc__)
