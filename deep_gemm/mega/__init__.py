"""Public PPU MegaMoE workspace and complete-call wrappers.

Matches the export organization of upstream ``deep_gemm.mega``. Quantized
operands use the PPU MXFP4 format; C++ validates tensors and runs the pipeline.
"""
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from .. import deep_gemm_cpp
from ..jit_kernels.utils import is_ppu1v5_device

__all__ = [
    "SymmBuffer", "get_symm_buffer_for_mega_moe", "transform_weights_for_mega_moe",
    "fp4_fp4_mega_moe", "mega_moe_pre_dispatch",
]

Tile = Tuple[int, int, int, int, int, int]


def _check_activation(activation: str) -> None:
    if activation not in ("swiglu", "step", "situ"):
        raise ValueError(f"unsupported precision/activation combination for W4A4: {activation}")


class MegaMoEGemm2Metadata(NamedTuple):
    m_rows: torch.Tensor
    valid_m_rows: torch.Tensor
    m_offsets: torch.Tensor
    src_m_rows: torch.Tensor
    src_m_offsets: torch.Tensor
    token_meta: torch.Tensor


def _tile_config(value, name):
    if value is None:
        return None
    if (not isinstance(value, (tuple, list)) or len(value) != 6 or
            any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError(f"{name} must contain six positive integers: "
                         "(BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, stages)")
    return tuple(value)


class SymmBuffer:
    """Reusable symmetric workspace for PPU W4A4 routed experts.

    ``intermediate_hidden`` is I; GEMM1 has 2I output columns. ``x`` is packed
    uint8 MXFP4 [capacity, H/2], ``x_sf`` is uint8 UE8M0 [capacity, H/32],
    ``topk_idx`` is int64 [capacity, topk], and ``topk_weights`` is FP32 with
    the same shape. Fill these views directly or use ``mega_moe_pre_dispatch``.
    Only the prefix matching the output's token count is consumed.

    All ranks allocate matching workspaces. Scratch, metadata, combine storage
    and communication state are owned here and recycled by the complete call.
    Use one stream per workspace; destroy it after outstanding work completes.
    """

    def __init__(self, group: dist.ProcessGroup, num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int, *,
                 mma_type: str = "fp4xfp4", activation: str = "swiglu",
                 gemm1_block_n: Optional[int] = None, gemm1_block_k: Optional[int] = None,
                 config_override: Optional[Dict[str, int]] = None,
                 gemm1_config: Optional[Tile] = None, gemm2_config: Optional[Tile] = None):
        import torch.distributed._symmetric_memory as symm_mem

        if not is_ppu1v5_device():
            raise RuntimeError("MegaMoE requires PPU1.5")
        if mma_type != "fp4xfp4":
            raise ValueError("PPU MegaMoE requires mma_type='fp4xfp4' (MXFP4)")
        _check_activation(activation)
        gemm1_config = _tile_config(gemm1_config, "gemm1_config")
        gemm2_config = _tile_config(gemm2_config, "gemm2_config")
        # C++ owns layout, validation and selection. Python only allocates
        # storage and retains its views, following upstream SymmBuffer.
        total, slice_input_buffers = deep_gemm_cpp.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type, activation,
            gemm1_block_n, gemm1_block_k, config_override, gemm1_config, gemm2_config)
        try:
            symm_mem.enable_symm_mem_for_group(group.group_name)
        except Exception:
            pass
        self.mma_type = mma_type
        self.activation = activation
        self.group = group
        self.num_ranks = group.size()
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden

        self._buffer = symm_mem.empty(total, dtype=torch.uint8, device='cuda')
        self._buffer.zero_()
        self._handle = symm_mem.rendezvous(self._buffer, group=group)
        group.barrier()
        torch.cuda.synchronize()

        ptrs = [self._handle.get_buffer(r, (total, ), torch.uint8, storage_offset=0).data_ptr()
                for r in range(self.num_ranks)]
        self._buf_ptrs = torch.tensor(ptrs, dtype=torch.int64, device='cpu')
        (self.x, self.x_sf, self.topk_idx, self.topk_weights,
         self.expert_recv_count, self.token_src_metadata,
         self.recv_count_sum, self.dispatch_arrival_count,
         m_rows, valid_m_rows, m_offsets, src_m_rows, src_m_offsets, token_meta,
         self._dispatch_counts, self._state) = slice_input_buffers(self._buffer)
        self._metadata = MegaMoEGemm2Metadata(
            m_rows, valid_m_rows, m_offsets, src_m_rows, src_m_offsets, token_meta)
        self.num_pool = token_meta.numel()
        # Reserve scales for the complete pool, independent of runtime tiles.
        self._sfa_scratch = torch.empty(self.num_pool * (hidden // 64), dtype=torch.uint16, device='cuda')
        self._init_pipeline()

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    @property
    def buf_ptrs(self) -> torch.Tensor:
        return self._buf_ptrs

    def get_gemm2_metadata(self) -> MegaMoEGemm2Metadata:
        """Persistent views populated by dispatch/GEMM1 on every invocation.

        GEMM2 padding follows the selected tile, including an explicit
        ``gemm2_config`` override. Views are ready on the same stream after
        GEMM1, including graph replay and zero-token invocations.

        Packed int64 routes are indexed by m_offsets.
        W4A4 activations and scales retain the layout indexed by src_m_offsets.
        """
        return self._metadata

    def destroy(self) -> None:
        self._destroy_pipeline()
        self._handle = None
        self._buffer = None
        self._buf_ptrs = None
        self._sfa_scratch = None
        self._state = None
        for name in ("x", "x_sf", "topk_idx", "topk_weights", "expert_recv_count",
                     "token_src_metadata", "recv_count_sum", "dispatch_arrival_count"):
            setattr(self, name, None)
        self.group = None

    def _init_pipeline(self):
        import torch.distributed._symmetric_memory as symm_mem

        self.rank = self.group.rank()
        self._topk_weights = self.topk_weights
        pool, width = self.num_pool, self.intermediate_hidden
        self._l2_acts = torch.empty(pool, width // 2, dtype=torch.uint8, device="cuda")
        self._l2_acts_sf = torch.empty(width // 64, pool, dtype=torch.uint16, device="cuda").t()
        groups = self.num_experts // self.num_ranks
        # The grouped scheduler uses row counts directly below 128 experts.
        # Otherwise reserve four int32 values per block plus expert boundaries,
        # using the smallest supported M tile (16) so tuning can change on reuse.
        self._gemm2_block_info = (torch.empty(4 * (1 + groups + (pool + 15) // 16),
                                             dtype=torch.int32, device="cuda")
                                  if groups >= 128 else None)
        self._combine = symm_mem.empty(self.num_topk * self.num_max_tokens_per_rank * self.hidden,
                                        dtype=torch.bfloat16, device="cuda")
        self._combine_handle = symm_mem.rendezvous(self._combine, group=self.group)
        pointers = [self._combine_handle.get_buffer(r, (self._combine.numel(),), torch.bfloat16).data_ptr()
                    for r in range(self.num_ranks)]
        self._combine_offsets = torch.tensor([p - pointers[self.rank] for p in pointers], dtype=torch.int64,
                                             device="cpu")
        self._barrier_flag = symm_mem.empty(1, dtype=torch.int32, device="cuda")
        self._barrier_handle = symm_mem.rendezvous(self._barrier_flag, group=self.group)
        self._barrier_pointers = torch.tensor([
            self._barrier_handle.get_buffer(r, (1,), torch.int32).data_ptr()
            for r in range(self.num_ranks)], dtype=torch.int64, device="cuda")
        self._barrier_done = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._barrier_flag.zero_()
        torch.cuda.synchronize()
        self.group.barrier()

    def _destroy_pipeline(self):
        for name in ("_l2_acts", "_l2_acts_sf", "_metadata",
                     "_gemm2_block_info",
                     "_topk_weights", "_dispatch_counts", "_combine", "_combine_handle", "_combine_offsets",
                     "_barrier_flag", "_barrier_handle", "_barrier_pointers", "_barrier_done"):
            setattr(self, name, None)


def get_symm_buffer_for_mega_moe(
        group: dist.ProcessGroup, num_experts: int, num_max_tokens_per_rank: int,
        num_topk: int, hidden: int, intermediate_hidden: int, *,
        mma_type: str = "fp4xfp4", activation: str = "swiglu",
        gemm1_block_n: Optional[int] = None, gemm1_block_k: Optional[int] = None,
        config_override: Optional[Dict[str, int]] = None,
        gemm1_config: Optional[Tile] = None, gemm2_config: Optional[Tile] = None,
) -> SymmBuffer:
    """Construct a SymmBuffer; dimensions and defaults match its constructor.

    Tiles contain (BM, BN, BK, WM, WN, stages); BK counts packed FP4 bytes.
    Explicit tiles override measured tuning. Otherwise the complete call uses
    actual tokens with heuristic fallback. UltraEP counts physical expert slots
    and requires remapped IDs plus matching master/replica weights in both stages.
    """
    return SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
        mma_type=mma_type, activation=activation, gemm1_block_n=gemm1_block_n,
        gemm1_block_k=gemm1_block_k, config_override=config_override,
        gemm1_config=gemm1_config, gemm2_config=gemm2_config)


def _interleave_weights(tensor: torch.Tensor, gran: int = 8) -> torch.Tensor:
    experts, n, k = tensor.shape
    return (tensor.reshape(experts, 2, n // (2 * gran), gran, k)
            .transpose(1, 2).contiguous().reshape(experts, n, k))


def transform_weights_for_mega_moe(
        l1_weights: Tuple[torch.Tensor, torch.Tensor],
        l2_weights: Tuple[torch.Tensor, torch.Tensor], activation: str = "swiglu",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Pack logical [gate; up] MXFP4 weights once, without requantization.

    Packed uint8 weights are [E, N, K/2], with uint8 UE8M0 scales [E, N, K/32].
    Use the returned pairs only with this backend. NVFP4 E4M3 scales are unsupported.
    """
    _check_activation(activation)
    if not is_ppu1v5_device():
        raise RuntimeError("MegaMoE requires PPU1.5")

    def validate_pair(name, pair):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"{name} must be a (packed MXFP4 weight, raw UE8M0 scale) pair")
        if not all(isinstance(t, torch.Tensor) for t in pair):
            raise TypeError(f"{name} weights and scales must be tensors")
        weight, scale = pair
        if weight.ndim != 3 or scale.ndim != 3:
            raise ValueError(f"{name} weights and scales must be rank-3 tensors")
        if any(t.dtype != torch.uint8 or not t.is_cuda or not t.is_contiguous() for t in pair):
            raise ValueError(f"{name} requires contiguous uint8 weights and raw UE8M0 scales on a PPU")
        if weight.device != scale.device:
            raise ValueError(f"{name} weights and scales must share a device")
        e, n, k = weight.shape
        if scale.shape != (e, n, k // 16):
            raise ValueError(f"{name} scales must have shape [local_experts, N, K/32]")
        return weight, scale

    l1_weight, l1_scale = validate_pair("l1_weights", l1_weights)
    l2_weight, l2_scale = validate_pair("l2_weights", l2_weights)
    experts, n, _ = l1_weight.shape
    intermediate, hidden = n // 2, l2_weight.shape[1]
    if (experts <= 0 or intermediate <= 0 or intermediate % 128 != 0 or
            hidden <= 0 or hidden % 512 != 0):
        raise ValueError("unsupported MegaMoE weight dimensions")
    if (l1_weight.shape != (experts, 2 * intermediate, hidden // 2) or
            l2_weight.shape != (experts, hidden, intermediate // 2)):
        raise ValueError("MegaMoE packed weights must have shapes [E, 2*I, H/2] and [E, H, I/2]")
    if l1_weight.device != l2_weight.device:
        raise ValueError("all MegaMoE weights and scales must share a device")

    l1_transformed = _interleave_weights(l1_weight)
    # Each 256-column N tile stores its UE8M0 scale pairs in K-major order.
    l1_scale = (_interleave_weights(l1_scale).view(torch.uint16)
                .reshape(experts, n // 256, 256, hidden // 64)
                .transpose(2, 3).contiguous().reshape(experts, n * hidden // 64))
    # GEMM2 consumes uint16 scale pairs in N-major order within each expert.
    l2_scale = l2_scale.view(torch.uint16).transpose(1, 2).contiguous().transpose(1, 2)
    return (l1_transformed, l1_scale), (l2_weight, l2_scale)


def _check_buffer(sym_buffer: SymmBuffer) -> None:
    if not isinstance(sym_buffer, SymmBuffer):
        raise TypeError("sym_buffer must be a SymmBuffer")
    if sym_buffer.buffer is None:
        raise ValueError("SymmBuffer has been destroyed")
    if sym_buffer.mma_type != "fp4xfp4":
        raise ValueError("PPU MegaMoE requires mma_type='fp4xfp4' (MXFP4)")


def mega_moe_pre_dispatch(
        hidden_states: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
        sym_buffer: SymmBuffer, num_tokens: Optional[int] = None,
) -> None:
    """Quantize BF16 inputs and stage routes into the four public input views.

    This optional PPU helper can be replaced by filling the views in a preceding
    operator. The complete call derives its actual token count from y.shape[0].
    """
    _check_buffer(sym_buffer)
    deep_gemm_cpp.mega_moe_pre_dispatch(
        hidden_states, topk_idx, topk_weights,
        sym_buffer.x, sym_buffer.x_sf, sym_buffer.topk_idx, sym_buffer.topk_weights, num_tokens)


def fp4_fp4_mega_moe(
        y: torch.Tensor, l1_weights: Tuple[torch.Tensor, torch.Tensor],
        l2_weights: Tuple[torch.Tensor, torch.Tensor], sym_buffer: SymmBuffer,
        cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
        recipe: Tuple[int, int, int] = (1, 1, 32),
        activation: str = "swiglu", activation_clamp: Optional[float] = None,
        fast_math: bool = True, *,
        l1_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        l2_weights_replicas: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        situ_beta: float = 4.0, situ_linear_beta: float = 25.0,
) -> None:
    """Write complete W4A4 routed MoE results into BF16 y [local_tokens, H].

    Inputs must already occupy the matching prefix of x/x_sf/topk_idx/topk_weights.
    Routing weights are applied once in combine. Every rank calls, including empty
    ranks. The library owns scratch and recycling; graph replay retains allocations
    and shapes. fast_math selects approximate GEMM1 activation math; False uses
    expf/division/tanhf without changing GEMM accumulation or MXFP4 formats.
    Common arguments follow SM90 fp8_mega_moe. The MXFP4 recipe is (1, 1, 32).
    Optional INT32 receive statistics have one element per local physical expert;
    each call adds valid routed assignments from all ranks, including zero-weight
    assignments. Initialize them once and retain them for graph replay; the library
    never resets these caller-owned cumulative statistics. SiTU coefficients and
    replica weights are keyword-only PPU extensions.
    """
    _check_buffer(sym_buffer)
    _check_activation(activation)
    if activation != sym_buffer.activation:
        raise ValueError("activation does not match SymmBuffer activation")
    metadata = sym_buffer.get_gemm2_metadata()
    deep_gemm_cpp.fp4_fp4_mega_moe(
        y, l1_weights, l2_weights, cumulative_local_expert_recv_stats,
        sym_buffer.buffer, sym_buffer.buf_ptrs, sym_buffer.group.rank(),
        sym_buffer.num_max_tokens_per_rank, sym_buffer.num_experts, sym_buffer.num_topk,
        recipe, activation, activation_clamp, fast_math,
        sym_buffer._state, sym_buffer._l2_acts, sym_buffer._l2_acts_sf, sym_buffer._sfa_scratch,
        metadata.m_rows, metadata.valid_m_rows, metadata.src_m_offsets, metadata.token_meta,
        sym_buffer._gemm2_block_info, sym_buffer._combine, sym_buffer._combine_offsets,
        sym_buffer._dispatch_counts, sym_buffer._topk_weights,
        sym_buffer._barrier_pointers, sym_buffer._barrier_done,
        l1_weights_replicas, l2_weights_replicas, situ_beta, situ_linear_beta)
