"""PPU1.5 NVFP4 W4A8 accuracy tests and offline scale absorption example.

Run: python tests/test_nvfp4.py
Or:  python -m pytest tests/test_nvfp4.py -q -s
"""
import json

import torch
import deep_gemm
from utils import is_ppu1v5_device


def _check_nvfp4(out, ref):
    expected = ref.to(torch.bfloat16)
    diff = (out.float() - expected.float()).square().sum()
    norm = expected.float().square().sum().clamp_min(1e-10)
    assert (diff / norm).item() < 2e-05
    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.25)


def _test_nvfp4_gemm(execution):
    # Keep both pytest and the standalone runner's FP32 references free of TF32.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mode = (deep_gemm.PreprocessMode.W4A8_NVFP4 if execution == 'fused'
                else deep_gemm.PreprocessMode.W4A8_NVFP4_FP8)
        lut = torch.tensor(
            [0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device='cuda')
        for batch in (1, 17, 128, 257, 4096):
            for kind in ('dense', 'nopad', 'masked', 'fused'):
                torch.manual_seed(42)
                e = 1 if kind == 'dense' else 4
                # Small cases exercise N/K tails; large ones reuse K pipeline stages.
                n, k = (192, 160) if batch < 128 else (256, 1024)
                if execution == 'dequantize' and kind == 'fused':
                    n, k = 256, 256  # Mainline FP8 routing requires N/K % 128 == 0.
                a_group = 0 if batch == 1 else 128
                b_block = 0 if execution == 'fused' and batch == 1 else 128
                counts = [0, batch // 4, batch // 3, batch - batch // 4 - batch // 3]
                if kind == 'masked':
                    counts[-1] = batch
                rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
                codes = torch.randint(16, (e, n, k), device='cuda', dtype=torch.uint8)
                native = (torch.rand(e, n, k // 16, device='cuda') + .25).to(torch.float8_e4m3fn)

                def prepare():
                    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
                    return deep_gemm.preprocess_weights_and_scales(mode, False, packed, native)

                rhs = prepare()
                a_shape = (e, batch, k) if kind == 'masked' else (batch, k)
                a = torch.randn(a_shape, device='cuda').to(torch.float8_e4m3fn)
                sa = torch.rand((*a_shape[:-1], (k + 127) // 128 if a_group else 1), device='cuda') + .25
                sb_shape = (e, (n + 127) // 128, (k + 127) // 128) if b_block else (e, n)
                sb = torch.rand(sb_shape, device='cuda') + .25
                sb[:, 0] = 0  # Include zero outer weight scales.
                gs = torch.linspace(.5, 1.25, e, device='cuda')
                out_shape = (*a_shape[:-1], n) if kind != 'fused' else (batch * 2, n)
                out = torch.full(out_shape, 123., device='cuda', dtype=torch.bfloat16)
                workspace = (torch.empty((e, n, k), device='cuda', dtype=torch.float8_e4m3fn)
                             if execution == 'dequantize' else None)
                configs = None
                kwargs = dict(
                    lhs_scale_group_size=a_group, rhs_scales=sb[0] if kind == 'dense' else sb,
                    rhs_scale_block_size=b_block, weight_global_scale=gs[0] if kind == 'dense' else gs,
                    execution=execution, fp8_workspace=workspace[0] if kind == 'dense' and workspace is not None else workspace,
                )
                if kind == 'nopad':
                    indices = torch.repeat_interleave(torch.arange(e, device='cuda', dtype=torch.int32), rows)
                elif kind == 'fused':
                    # Expert zero is empty; every token selects two distinct experts.
                    routes = ((torch.arange(batch, device='cuda')[:, None]
                               + torch.arange(2, device='cuda')[None, :]) % (e - 1) + 1).int()
                    routing_weight = workspace if workspace is not None else rhs[0]
                    if execution == 'fused':
                        configs = deep_gemm.get_gemm_fp8_nvfp4_config(
                            'MGroupedFused', (batch * 2 + e - 1) // e, n, k,
                            lhs_scale_group_size=a_group, rhs_scale_block_size=b_block)
                    configs, aligned_rows, experts, tokens, blocks, _, _ = deep_gemm.moe_align_block_size(
                        a, routing_weight, routes, False, config=configs)
                if execution == 'fused' and batch == 257 and kind in ('nopad', 'masked'):
                    # One representative shared-decode configuration, including padded rows.
                    configs = (deep_gemm.get_num_sms(), 128, 128, 128, 64, 32, 128, 3)

                def run():
                    lhs = (a, sa)
                    if kind == 'dense':
                        deep_gemm.gemm_fp8_nvfp4_bf16_nt(
                            lhs, tuple(x[0] for x in rhs), out, **kwargs)
                    elif kind == 'nopad':
                        deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
                            lhs, rhs, out, indices, m_rows=rows, configs=configs, **kwargs)
                    elif kind == 'masked':
                        deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_masked(
                            lhs, rhs, out, rows, batch, configs=configs, **kwargs)
                    else:
                        deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_fused(
                            lhs, rhs, out, aligned_rows, experts, tokens, blocks, configs, **kwargs)

                def check():
                    ar = a.float() * sa.repeat_interleave(a_group or k, -1)[..., :k]
                    weight = lut[codes.long()] * native.float().repeat_interleave(16, -1)
                    if b_block:
                        # Current blockwise execution rounds native NVFP4 values to FP8.
                        weight = weight.clamp(-448, 448).to(torch.float8_e4m3fn).float()
                        weight *= sb.repeat_interleave(128, -2).repeat_interleave(128, -1)[:, :n, :k]
                    else:
                        weight *= sb[..., None]
                    weight *= gs[:, None, None]
                    if kind == 'dense':
                        _check_nvfp4(out, ar @ weight[0].t())
                    elif kind == 'masked':
                        for ex, count in enumerate(counts):
                            if count:
                                _check_nvfp4(out[ex, :count], ar[ex, :count] @ weight[ex].t())
                            assert torch.all(out[ex, count:] == 123.)
                    elif kind == 'nopad':
                        offsets = [0]
                        for count in counts:
                            offsets.append(offsets[-1] + count)
                        ref = torch.cat([ar[lo:hi] @ weight[ex].t()
                                         for ex, (lo, hi) in enumerate(zip(offsets, offsets[1:])) if hi > lo])
                        _check_nvfp4(out, ref)
                    else:
                        token_ids = torch.arange(batch, device='cuda').repeat_interleave(2)
                        ref = torch.cat([ar[token_ids[routes.flatten() == ex]] @ weight[ex].t()
                                         for ex in range(1, e)])
                        _check_nvfp4(out, ref)
                    if workspace is not None:
                        decoded = (lut[codes.long()] * native.float().repeat_interleave(16, -1))
                        expected = decoded.clamp(-448, 448).to(torch.float8_e4m3fn)
                        torch.testing.assert_close(workspace.float(), expected.float(), rtol=0, atol=0)

                print(f'NVFP4 {execution}: {kind}, batch={batch}, N={n}, K={k}', flush=True)
                if execution == 'fused' and kind == 'dense' and batch == 17:
                    # Binary K-warp reduction requires power-of-two group counts.
                    invalid = (deep_gemm.get_num_sms(), 16, 64, 96, 16, 32, 32, 2)
                    try:
                        deep_gemm.gemm_fp8_nvfp4_bf16_nt(
                            (a, sa), tuple(x[0] for x in rhs), out, configs=invalid, **kwargs)
                    except RuntimeError as error:
                        assert 'power of two' in str(error)
                    else:
                        raise AssertionError('Non-power-of-two K-warp reduction must be rejected')
                run()
                check()
                if batch == 17 and kind == 'nopad':
                    # One graph replay per execution mode verifies that weights and scales refresh.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    codes = torch.randint(16, codes.shape, device='cuda', dtype=torch.uint8)
                    native = (native.float() * .75).to(native.dtype)
                    updated = prepare()
                    for dst, src in zip(rhs, updated):
                        dst.copy_(src)
                    sa.mul_(.75)
                    sb.add_(.125)
                    gs.mul_(.875)
                    graph.replay()
                    check()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def test_nvfp4_decode():
    """Fused dense/grouped GEMMs across small, aligned, partial and large batches."""
    _test_nvfp4_gemm('fused')


def test_nvfp4_prefill():
    """Per-call weight decoding followed by FP8 dense/grouped GEMMs."""
    _test_nvfp4_gemm('dequantize')


def _decode_nvfp4(codes, scales):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                       device=codes.device)
    return lut[codes.long()] * scales.float().repeat_interleave(16, -1)


def _cast_fp8(values):
    # Match the existing kernel's finite saturation, including out-of-range input.
    return values.clamp(-448, 448).to(torch.float8_e4m3fn)


def _expand_block_scale(scale, n, k):
    return scale.repeat_interleave(128, -2).repeat_interleave(128, -1)[..., :n, :k]


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-20)).item()


def _absorb_fp8_scale(codes, native_scale):
    """Return normalized K16 E4M3 scales and restoring 128x128 FP32 scales.

    Global scale is kept separate and passed unchanged to both GEMM paths.
    Power-of-two normalization preserves native E4M3 scales except at their
    representable range boundaries. Small scales can still underflow; the
    comparison measures that loss against the original NVFP4 values.
    """
    e, n, k = codes.shape
    values = _decode_nvfp4(codes, native_scale)
    amax = values.abs().reshape(e, n // 128, 128, k // 128, 128).amax(dim=(2, 4))
    scale_amax = native_scale.float().reshape(e, n // 128, 128, k // 128, 8).amax(dim=(2, 4))
    # Also keep the normalized K16 scales themselves within E4M3 range.
    bound = torch.maximum(amax, scale_amax)
    rhs_scales = torch.exp2(torch.ceil(torch.log2((bound / 448).clamp_min(1e-30))))
    rhs_scales = torch.where(bound == 0, torch.ones_like(rhs_scales), rhs_scales)
    divisor = rhs_scales.repeat_interleave(128, -2).repeat_interleave(8, -1)
    normalized = (native_scale.float() / divisor).to(torch.float8_e4m3fn)
    assert torch.isfinite(normalized.float()).all()
    assert _decode_nvfp4(codes, normalized).abs().max() <= 448
    return normalized, rhs_scales


def test_nvfp4_offline_scale():
    """Compare offline absorption and direct casts against the same NVFP4 weights.

    Printed relative L2 errors are norm(actual - reference) / norm(reference).
    The NVFP4 reference uses the original K16/global scales in FP32; the FP8
    reference includes each method's conversion and outer scales.
    """
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        _test_nvfp4_offline_scale()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _test_nvfp4_offline_scale():
    e, n, k = 3, 256, 256
    for profile in ('small', 'large', 'wide'):
        torch.manual_seed(42)
        codes = torch.randint(16, (e, n, k), device='cuda', dtype=torch.uint8)
        samples = torch.rand((e, n, k // 16), device='cuda')
        if profile == 'small':
            scales = samples + .25
        elif profile == 'large':
            scales = 128 + 320 * samples
        else:
            scales = torch.exp2(-9 + (9 + 8.807354922) * samples)
        native = scales.to(torch.float8_e4m3fn)
        if profile != 'small':
            native[..., 0] = 448
            codes[..., 0] = 7  # q=6 exposes the direct cast's range limit.
        global_scale = torch.exp2(torch.linspace(-12, -6, e, device='cuda'))
        original = _decode_nvfp4(codes, native)
        target = original * global_scale[:, None, None]
        folded, restore = _absorb_fp8_scale(codes, native)
        methods = [('direct_cast', native, torch.ones_like(restore)),
                   ('offline_absorb', folded, restore)]
        weight_errors = {}
        for method, kernel_scales, rhs_scales in methods:
            pre_cast = _decode_nvfp4(codes, kernel_scales)
            # E2M1 x E4M3 fits BF16 exactly before either FP8 cast.
            torch.testing.assert_close(pre_cast, pre_cast.bfloat16().float(), rtol=0, atol=0)
            w8 = _cast_fp8(pre_cast)
            restored = w8.float() * _expand_block_scale(rhs_scales, n, k) * global_scale[:, None, None]
            weight_errors[method] = _relative_l2(restored, target)
            saturated = int((pre_cast.abs() > 448).sum().item())
            if method == 'offline_absorb':
                assert saturated == 0 and weight_errors[method] < .05
                if profile == 'small':
                    torch.testing.assert_close(restored, direct_restored, rtol=0, atol=0)
            else:
                direct_restored = restored
            print(json.dumps({
                'profile': profile, 'method': method,
                'weight_vs_nvfp4_rel_l2': weight_errors[method],
                'saturated_values': saturated,
                'nonzero_to_zero': int(((w8.float() == 0) & (original != 0)).sum().item()),
            }), flush=True)
            packed = codes[..., ::2] | (codes[..., 1::2] << 4)
            prepared = {
                execution: deep_gemm.preprocess_weights_and_scales(mode, False, packed, kernel_scales)
                for execution, mode in [('fused', deep_gemm.PreprocessMode.W4A8_NVFP4),
                                        ('dequantize', deep_gemm.PreprocessMode.W4A8_NVFP4_FP8)]
            }
            for batch in (1, 128, 4096):
                torch.manual_seed(100 + batch)
                a = torch.randn((batch, k), device='cuda').to(torch.float8_e4m3fn)
                sa = torch.rand((batch, k // 128), device='cuda') + .25
                ar = a.float() * sa.repeat_interleave(128, -1)
                counts = [0, batch // 3, batch - batch // 3]
                rows = torch.tensor(counts, device='cuda', dtype=torch.int32)
                indices = torch.repeat_interleave(torch.arange(e, device='cuda', dtype=torch.int32), rows)
                for kind in ('dense', 'nopad'):
                    def matmul(weight):
                        if kind == 'dense':
                            return ar @ weight[0].t()
                        split = counts[1]
                        return torch.cat([ar[:split] @ weight[1].t(), ar[split:] @ weight[2].t()])

                    reference = matmul(target)
                    fp8_reference = matmul(restored)
                    record = {'profile': profile, 'method': method, 'batch': batch, 'kind': kind,
                              'fp8_reference_vs_nvfp4_rel_l2': _relative_l2(fp8_reference, reference)}
                    outputs = []
                    for execution in ('fused', 'dequantize'):
                        rhs = prepared[execution]
                        workspace = (torch.empty((e, n, k), device='cuda', dtype=torch.float8_e4m3fn)
                                     if execution == 'dequantize' else None)
                        kwargs = dict(
                            lhs_scale_group_size=128, rhs_scale_block_size=128,
                            rhs_scales=rhs_scales[0] if kind == 'dense' else rhs_scales,
                            weight_global_scale=global_scale[0] if kind == 'dense' else global_scale,
                            execution=execution,
                            fp8_workspace=workspace[0] if kind == 'dense' and workspace is not None else workspace,
                        )
                        out = torch.empty((batch, n), device='cuda', dtype=torch.bfloat16)
                        if kind == 'dense':
                            deep_gemm.gemm_fp8_nvfp4_bf16_nt(
                                (a, sa), tuple(x[0] for x in rhs), out, **kwargs)
                        else:
                            deep_gemm.m_grouped_gemm_fp8_nvfp4_bf16_nt_nopad(
                                (a, sa), rhs, out, indices, m_rows=rows, **kwargs)
                        kernel_error = _relative_l2(out, fp8_reference)
                        output_error = _relative_l2(out, reference)
                        assert kernel_error < .006, (profile, method, batch, kind, execution, kernel_error)
                        if method == 'offline_absorb':
                            assert output_error < .06, (profile, batch, kind, execution, output_error)
                        record[execution + '_vs_fp8_reference_rel_l2'] = kernel_error
                        record[execution + '_vs_nvfp4_rel_l2'] = output_error
                        if workspace is not None:
                            decoded = workspace[0] if kind == 'dense' else workspace
                            expected = w8[0] if kind == 'dense' else w8
                            torch.testing.assert_close(decoded.float(), expected.float(), rtol=0, atol=0)
                        outputs.append(out)
                    record['fused_vs_dequantize_rel_l2'] = _relative_l2(outputs[0], outputs[1])
                    assert record['fused_vs_dequantize_rel_l2'] < .006
                    print(json.dumps(record), flush=True)
        if profile != 'small':
            assert weight_errors['offline_absorb'] < weight_errors['direct_cast']


if __name__ == '__main__':
    if is_ppu1v5_device():
        test_nvfp4_decode()
        test_nvfp4_prefill()
        test_nvfp4_offline_scale()
    else:
        print('Skipping NVFP4: requires PPU1.5')
