"""Whole-weight MXFP4 -> blockwise FP8 conversion and prefill execution."""
import pytest
import torch
import deep_gemm as dg
from deep_gemm.testing.numeric import calc_diff


def prepare(packed, scales):
    return dg.preprocess_weights_and_scales(dg.PreprocessMode.W4A8_MXFP4_FP8, False, packed, scales)


def reference(packed, scales):
    """CPU reference, normalizing in float64 before the final E4M3 cast."""
    packed, scales = packed.cpu(), scales.cpu()
    e, n, half_k = packed.shape
    k = half_k * 2
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    mag = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)[codes & 7]
    values = torch.where((codes & 8).bool(), -mag, mag)
    maxima = scales.reshape(e, n // 128, 128, k // 128, 4).amax((2, 4)).int()
    expanded = maxima.repeat_interleave(128, 1).repeat_interleave(4, 2)
    normalized = values * torch.exp2((scales.int() - expanded + 6).double()).repeat_interleave(32, -1)
    outer = torch.exp2((maxima - 133).double()).float()
    return normalized.float().to(torch.float8_e4m3fn), outer


@pytest.mark.parametrize("low,high", [(108, 127), (109, 135), (1, 6), (249, 254)])
def test_mxfp4_fp8_decode(low, high):
    e, n, k = 2, 256, 512
    codes = (torch.arange(e * n * k).reshape(e, n, k) % 16).to(torch.uint8).cuda()
    packed = (codes[..., ::2] | (codes[..., 1::2] << 4)).contiguous()
    raw = (torch.arange(e * n * k // 32).reshape(e, n, k // 32) % (high - low + 1) + low).to(torch.uint8).cuda()
    # Include all positive/negative E2M1 codes and all K32 scale boundaries.
    prepared = prepare(packed, raw)
    assert prepared[0].data_ptr() == packed.data_ptr()  # Share the fused path's packed weights.
    workspace = torch.empty((e, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    dg.dequantize_mxfp4_fp8(prepared[:2], workspace)
    expected, outer = reference(packed, raw)
    assert torch.equal(workspace.view(torch.uint8).cpu(), expected.view(torch.uint8))
    assert torch.equal(prepared[2].cpu(), outer)
    dense = torch.empty_like(workspace[0])
    dg.dequantize_mxfp4_fp8(tuple(t[0] for t in prepared[:2]), dense)
    assert torch.equal(dense.view(torch.uint8), workspace[0].view(torch.uint8))
    if (low, high) == (108, 127):
        assert prepared[1].unique().numel() == 20


def test_mxfp4_prefill_gemm_and_graph():
    torch.manual_seed(3114)
    counts = [0, 1, 17, 137]
    e, n, k, capacity = 4, 256, 384, 160
    packed = torch.randint(256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    raw = torch.randint(118, 123, (e, n, k // 32), dtype=torch.uint8, device="cuda")
    prepared = prepare(packed, raw)
    workspace = torch.empty((e, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    expected_w, expected_sf = reference(packed, raw)
    # This four-exponent spread is exactly representable by blockwise E4M3.
    weight = expected_w.float() * expected_sf.repeat_interleave(128, 1).repeat_interleave(128, 2)
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    cpu = packed.cpu()
    code = torch.stack((cpu & 15, cpu >> 4), -1).flatten(-2).long()
    original = lut[code] * torch.exp2(raw.cpu().float() - 127).repeat_interleave(32, -1)
    assert torch.equal(weight, original)
    rows = torch.tensor(counts, device="cuda", dtype=torch.int32)
    ids = torch.repeat_interleave(torch.arange(e, device="cuda", dtype=torch.int32), rows.long())
    x = torch.randn(sum(counts), k, device="cuda", dtype=torch.bfloat16)
    flat = dg.per_token_cast_to_fp8(x)
    lhs = (flat[0], dg.get_mn_major_tma_aligned_tensor(flat[1]))
    out = torch.empty(sum(counts), n, device="cuda", dtype=torch.bfloat16)

    def execute():
        dg.dequantize_mxfp4_fp8(prepared[:2], workspace)
        dg.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(lhs, (workspace, prepared[2]), out, ids, rows)

    execute()
    torch.cuda.synchronize()
    a = flat[0].float().cpu() * flat[1].cpu().repeat_interleave(128, -1)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            expected = (a[start:start + count] @ original[expert].T).bfloat16()
            assert calc_diff(out[start:start + count].cpu(), expected) < 1e-4
        start += count

    masked_a = torch.zeros(e, capacity, k, device="cuda", dtype=torch.float8_e4m3fn)
    masked_s = torch.zeros(e, capacity, k // 128, device="cuda")
    start = 0
    for expert, count in enumerate(counts):
        masked_a[expert, :count].copy_(flat[0][start:start + count])
        masked_s[expert, :count].copy_(flat[1][start:start + count])
        start += count
    masked_out = torch.full((e, capacity, n), -987., device="cuda", dtype=torch.bfloat16)
    dg.dequantize_mxfp4_fp8(prepared[:2], workspace)
    dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
        (masked_a, dg.get_mn_major_tma_aligned_tensor(masked_s)), (workspace, prepared[2]),
        masked_out, rows, 39)
    start = 0
    for expert, count in enumerate(counts):
        assert torch.equal(masked_out[expert, :count], out[start:start + count])
        assert (masked_out[expert, count:] == -987.).all()
        start += count

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    graph.replay()
    torch.cuda.synchronize()
    saved = out.clone()
    prepared[0].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert (workspace.float() == 0).all()
    assert (out == 0).all() and (saved != 0).any()


def test_mxfp4_fp8_invalid_operands():
    packed = torch.zeros(1, 128, 64, device="cuda", dtype=torch.uint8)
    raw = torch.full((1, 128, 4), 127, device="cuda", dtype=torch.uint8)
    prepared = prepare(packed, raw)
    workspace = torch.empty(1, 128, 128, device="cuda", dtype=torch.float8_e4m3fn)
    for code in (0, 255):
        bad = raw.clone()
        bad[0, 0, 0] = code
        with pytest.raises(RuntimeError, match="exponent codes"):
            prepare(packed, bad)
    with pytest.raises(RuntimeError, match="N%128"):
        prepare(packed[:, :64], raw[:, :64])
    with pytest.raises(RuntimeError, match="fuse_activation=False"):
        dg.preprocess_weights_and_scales(dg.PreprocessMode.W4A8_MXFP4_FP8, True, packed, raw)
    with pytest.raises(RuntimeError, match="prepared weights/indices"):
        dg.dequantize_mxfp4_fp8((prepared[0].to(torch.int64), prepared[1]), workspace)
    with pytest.raises(RuntimeError, match="PPU tensors"):
        dg.dequantize_mxfp4_fp8(tuple(t.cpu() for t in prepared[:2]), workspace.cpu())
    storage = torch.empty(workspace.numel(), device="cuda", dtype=torch.uint8)
    alias = storage[:packed.numel()].reshape(prepared[0].shape)
    with pytest.raises(RuntimeError, match="must not overlap"):
        dg.dequantize_mxfp4_fp8((alias, prepared[1]), storage.view(torch.float8_e4m3fn).reshape(workspace.shape))
    unaligned = torch.empty(prepared[0].numel() + 1, device="cuda", dtype=torch.uint8)[1:].reshape(prepared[0].shape)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        dg.dequantize_mxfp4_fp8((unaligned, prepared[1]), workspace)
    aligned = prepare(unaligned, raw)
    assert aligned[0].data_ptr() % 16 == 0
