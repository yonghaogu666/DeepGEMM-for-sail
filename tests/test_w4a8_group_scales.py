"""GPU regressions for group-128 activation scales; requires a rebuilt PPU library."""

import pytest
import torch
import deep_gemm

from quantization import decode_weight
from deep_gemm.testing.numeric import calc_diff


def test_activation_group_sizes():
    # Zero denotes per-row scaling; 128 denotes consecutive K128 groups.
    assert deep_gemm.get_w4a8_activation_group_sizes() == [0, 128]


def operands(shape, n):
    e, m, k = shape
    values = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
    # Different scales along both M and K expose epilogue-only scaling bugs.
    scales = torch.exp2(torch.randint(-5, 5, (e, m, k // 128), device="cuda").float())
    packed = torch.randint(256, (e, n, k // 2), device="cuda", dtype=torch.uint8)
    raw = torch.randint(109, 125, (e, n, k // 32), device="cuda", dtype=torch.uint8)
    rhs = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, packed, raw
    )
    a = values.float() * scales.repeat_interleave(128, -1)
    b = torch.stack([decode_weight(packed[i], raw[i]) for i in range(e)])
    return (values, scales), rhs, torch.bmm(a, b.transpose(-1, -2)).bfloat16()


@pytest.mark.parametrize("k", [128, 256, 384, 2048])
@pytest.mark.parametrize("warp_k", [64, 128])
@pytest.mark.parametrize("masked", [False, True])
def test_group128(k, warp_k, masked):
    torch.manual_seed(20261103)
    e, capacity, n = 4, 65, 768
    lhs, rhs, reference = operands((e, capacity, k), n)
    counts = [0, 1, 17, 65]
    rows = torch.tensor(counts, dtype=torch.int32, device="cuda")
    sms = deep_gemm.get_num_sms()
    expand = 3 if not masked and warp_k == 128 and k <= 384 else 1
    config = (sms, 32, 256, 128, 32, 64, warp_k, 2, expand)
    if masked:
        out = torch.full_like(reference, -123.0)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
            lhs, rhs[:2], out, rows, 21, config, inv_scale=rhs[2]
        )
        for i, count in enumerate(counts):
            if count:
                assert calc_diff(out[i, :count], reference[i, :count]) < 0.0015
            assert (out[i, count:] == -123.0).all()
    else:
        a = torch.cat([lhs[0][i, :count] for i, count in enumerate(counts)])
        s = torch.cat([lhs[1][i, :count] for i, count in enumerate(counts)])
        ref = torch.cat([reference[i, :count] for i, count in enumerate(counts)])
        indices = torch.repeat_interleave(
            torch.arange(e, device="cuda", dtype=torch.int32), rows.long()
        )
        out = torch.empty_like(ref)
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
            (a, s), rhs[:2], out, indices, rows, config, inv_scale=rhs[2]
        )
        assert calc_diff(out, ref) < 0.0015


@pytest.mark.parametrize("k", [128, 384, 2048])
@pytest.mark.parametrize("masked", [False, True])
def test_group128_narrow_warp(k, masked):
    torch.manual_seed(20261106)
    e, capacity, n = 4, 129, 768
    lhs, rhs, _ = operands((e, capacity, k), n)
    values, scales = lhs
    # Non-power-of-two and zero scales ensure no exponent-only shortcut is used.
    scales.mul_(torch.rand_like(scales) + 0.25)
    scales[:, ::7, ::2] = 0
    counts = [0, 1, 63, 129]
    rows = torch.tensor(counts, dtype=torch.int32, device="cuda")
    sms = deep_gemm.get_num_sms()
    narrow = (sms, 64, 256, 128, 64, 32, 128, 2, 1)
    # Existing small-tile path is independently checked against FP32 above.
    reference_config = (sms, 32, 256, 128, 32, 64, 128, 2, 1)
    inverse = rhs[2] * 0.3
    if masked:
        out = torch.full((e, capacity, n), -123., device="cuda", dtype=torch.bfloat16)
        reference = torch.full_like(out, -123.)
        def run(dst, cfg):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
                (values, scales), rhs[:2], dst, rows, 48, cfg, inv_scale=inverse
            )
    else:
        values = torch.cat([values[i, :count] for i, count in enumerate(counts)])
        scales = torch.cat([scales[i, :count] for i, count in enumerate(counts)])
        indices = torch.repeat_interleave(torch.arange(e, device="cuda", dtype=torch.int32), rows.long())
        out = torch.full((sum(counts), n), -123., device="cuda", dtype=torch.bfloat16)
        reference = torch.full_like(out, -123.)
        def run(dst, cfg):
            deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
                (values, scales), rhs[:2], dst, indices, rows, cfg, inv_scale=inverse
            )
    def check_result():
        if masked:
            for i, count in enumerate(counts):
                if count:
                    assert calc_diff(out[i, :count], reference[i, :count]) < 0.0015
        else:
            assert calc_diff(out, reference) < 0.0015

    run(reference, reference_config)
    run(out, narrow)
    torch.cuda.synchronize()
    check_result()
    # The default selector must choose a usable group-scale configuration too.
    out.fill_(-123.)
    run(out, None)
    torch.cuda.synchronize()
    check_result()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out, narrow)
    scales.mul_(1.25)
    run(reference, reference_config)
    out.fill_(-123.)
    graph.replay()
    torch.cuda.synchronize()
    check_result()
    if masked:
        for i, count in enumerate(counts):
            assert (out[i, count:] == -123.).all()


def test_fused_group128():
    torch.manual_seed(20261103)
    e, m, n, k = 8, 17, 192, 384
    lhs, rhs, _ = operands((e, m, k), n)
    a, scales = lhs[0][0].contiguous(), lhs[1][0].contiguous()
    routes = torch.stack([torch.randperm(e, device="cuda")[:3] for _ in range(m)]).int()
    config, rows, experts, tokens, aligned, inv_perm, _ = (
        deep_gemm.moe_align_block_size(a, rhs[0], routes)
    )
    out = torch.empty(m * 3, n, device="cuda", dtype=torch.bfloat16)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_fused(
        (a, scales),
        rhs[:2],
        out,
        rows,
        experts,
        tokens,
        aligned,
        config,
        inv_scale=rhs[2],
    )
    # Compare the fused gather to the same grouped GEMM with explicit routing.
    order = routes.flatten().argsort(stable=True)
    flat_ids = routes.flatten()[order]
    token_ids = torch.arange(m, device="cuda").repeat_interleave(3)[order]
    sorted_a, sorted_s = a[token_ids].contiguous(), scales[token_ids].contiguous()
    ref = torch.empty_like(out)
    deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad(
        (sorted_a, sorted_s), rhs[:2], ref, flat_ids, rows, inv_scale=rhs[2]
    )
    # inv_perm maps the original token/top-k rows to the fused output rows.
    ref_original = torch.empty_like(ref)
    ref_original[order] = ref
    assert calc_diff(out[inv_perm.long()], ref_original) < 0.0015


@pytest.mark.parametrize("n,k", [(4096, 6144), (6144, 2048)])
def test_group128_masked_m32_graph(n, k):
    """The default M16-boundary selection covers both projection shapes."""
    torch.manual_seed(20261109)
    counts = [0, 1, 7, 15, 16, 17, 31, 41] * 6
    e, capacity = len(counts), max(counts) + 3
    lhs, rhs, _ = operands((e, capacity, k), n)
    values, scales = lhs
    scales.mul_(torch.rand_like(scales) + 0.25)
    scales[:, ::7, ::2] = 0
    rows = torch.tensor(counts, device="cuda", dtype=torch.int32)
    out = torch.full((e, capacity, n), -123., device="cuda", dtype=torch.bfloat16)
    reference = torch.full_like(out, -123.)
    m16 = (deep_gemm.get_num_sms(), 16, 256, 128, 16, 64, 128, 2, 1)

    def run(dst, config=None):
        deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
            (values, scales), rhs[:2], dst, rows, 16, config, inv_scale=rhs[2]
        )

    def check():
        for expert, count in enumerate(counts):
            if count:
                assert torch.isfinite(out[expert, :count]).all()
                assert calc_diff(out[expert, :count], reference[expert, :count]) < 0.0015
            assert (out[expert, count:] == -123.).all()

    run(reference, m16)
    run(out)
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out)
    # Exercise both M16 and M32 tail bodies with fresh device-side counts on replay.
    counts.reverse()
    rows.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
    scales.mul_(0.75)
    out.fill_(-123.)
    reference.fill_(-123.)
    run(reference, m16)
    graph.replay()
    torch.cuda.synchronize()
    check()
