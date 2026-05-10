from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only on CUDA builds with Triton.
    triton = None
    tl = None


def _triton_block_size(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _can_use_triton_matrix_kernel(x: Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.ndim == 2
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


if triton is not None:

    @triton.jit
    def _row_normalize_kernel(x, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        ptrs = x + row * n_cols + offs
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(vals * vals, axis=0)
        den = tl.sqrt(tl.maximum(ss, eps * eps))
        tl.store(ptrs, vals / den, mask=mask)

    @triton.jit
    def _column_normalize_kernel(
        x, n_rows: tl.constexpr, n_cols: tl.constexpr, eps: tl.constexpr, block: tl.constexpr
    ):
        col = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_rows
        ptrs = x + offs * n_cols + col
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(vals * vals, axis=0)
        den = tl.sqrt(tl.maximum(ss, eps * eps))
        tl.store(ptrs, vals / den, mask=mask)

    @triton.jit
    def _row_grad_project_kernel(weight, grad, n_cols: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_cols
        w_ptrs = weight + row * n_cols + offs
        g_ptrs = grad + row * n_cols + offs
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(w * g, axis=0)
        tl.store(g_ptrs, g - dot * w, mask=mask)

    @triton.jit
    def _column_grad_project_kernel(
        weight, grad, n_rows: tl.constexpr, n_cols: tl.constexpr, block: tl.constexpr
    ):
        col = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < n_rows
        w_ptrs = weight + offs * n_cols + col
        g_ptrs = grad + offs * n_cols + col
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(w * g, axis=0)
        tl.store(g_ptrs, g - dot * w, mask=mask)


def row_denominator(x: Tensor, eps: float) -> Tensor:
    """Return guarded L2 denominators for the last axis."""

    den = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    if eps > 0:
        den = den.clamp_min(eps)
    return den


def normalize_last_dim(x: Tensor, eps: float) -> Tensor:
    return x / row_denominator(x, eps)


@torch.no_grad()
def normalize_rows_(x: Tensor, eps: float = 1e-12) -> Tensor:
    if _can_use_triton_matrix_kernel(x):
        block = _triton_block_size(x.shape[1])
        _row_normalize_kernel[(x.shape[0],)](x, x.shape[1], eps, block)
        return x
    x.div_(row_denominator(x, eps))
    return x


@torch.no_grad()
def normalize_columns_(x: Tensor, eps: float = 1e-12) -> Tensor:
    if _can_use_triton_matrix_kernel(x):
        block = _triton_block_size(x.shape[0])
        _column_normalize_kernel[(x.shape[1],)](x, x.shape[0], x.shape[1], eps, block)
        return x
    den = torch.linalg.vector_norm(x, ord=2, dim=0, keepdim=True).clamp_min(eps)
    x.div_(den)
    return x


@torch.no_grad()
def project_row_grad_(weight: Tensor, grad: Tensor) -> None:
    if _can_use_triton_matrix_kernel(weight) and _can_use_triton_matrix_kernel(grad):
        block = _triton_block_size(weight.shape[1])
        _row_grad_project_kernel[(weight.shape[0],)](weight, grad, weight.shape[1], block)
        return
    grad.sub_((grad * weight).sum(dim=-1, keepdim=True) * weight)


@torch.no_grad()
def project_column_grad_(weight: Tensor, grad: Tensor) -> None:
    if _can_use_triton_matrix_kernel(weight) and _can_use_triton_matrix_kernel(grad):
        block = _triton_block_size(weight.shape[0])
        _column_grad_project_kernel[(weight.shape[1],)](
            weight, grad, weight.shape[0], weight.shape[1], block
        )
        return
    grad.sub_(weight * (grad * weight).sum(dim=0, keepdim=True))


def qkv_postprocess_from_carried(
    qkv: Tensor,
    rho: Tensor,
    cos: Tensor,
    sin: Tensor,
    qk_scale: Tensor,
    norm_eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert raw QKV projected from carried `Y` into attention inputs.

    `qkv` has shape `[B, T, 3, H, Dh]` and is computed from `Y`, not from
    `Y / rho`.

    Q/K do not consume `rho` directly because the positive row scalar cancels
    under Q/K normalization. The guard still uses `rho * eps`, matching the
    guarded normalization of `(Y W) / rho`. V must consume `rho`, because
    attention averages value vectors across tokens.
    """

    bsz, seq_len, _, n_head, head_dim = qkv.shape
    q_raw, k_raw, v_raw = qkv.unbind(dim=2)
    q_den = torch.linalg.vector_norm(q_raw, ord=2, dim=-1)
    k_den = torch.linalg.vector_norm(k_raw, ord=2, dim=-1)
    if norm_eps > 0:
        guard = rho.unsqueeze(-1) * norm_eps
        q_den = torch.maximum(q_den, guard)
        k_den = torch.maximum(k_den, guard)
    scale = qk_scale.view(1, 1, n_head, head_dim)
    if torch.is_grad_enabled():
        q = apply_rope(q_raw, cos, sin)
        k = apply_rope(k_raw, cos, sin)
        q = (q / q_den.unsqueeze(-1)) * scale
        k = (k / k_den.unsqueeze(-1)) * scale
        v = v_raw / rho.view(bsz, seq_len, 1, 1)
    else:
        q = apply_rope_inplace(q_raw, cos, sin)
        k = apply_rope_inplace(k_raw, cos, sin)
        q.div_(q_den.unsqueeze(-1)).mul_(scale)
        k.div_(k_den.unsqueeze(-1)).mul_(scale)
        v = v_raw.div_(rho.view(bsz, seq_len, 1, 1))
    return q, k, v


def carried_up_gate_swiglu(
    y: Tensor,
    rho: Tensor,
    weight: Tensor,
    s_u: Tensor,
    s_gate: Tensor,
    *,
    sqrt_d: float,
    scale_u_by_sqrt_d: bool,
) -> Tensor:
    """MLP up/gate path from carried state.

    The current portable PyTorch kernel materializes `Y / rho` before the
    bias-free linear projection, matching the code-level semantics exactly. A
    CUDA/Triton backend can replace this helper with a row-scaled GEMM epilogue
    that never writes `Y / rho`, while preserving the same function signature.
    """

    mlp_in = y / rho.unsqueeze(-1)
    u, gate = F.linear(mlp_in, weight).chunk(2, dim=-1)
    u = u * s_u.view(1, 1, -1)
    if scale_u_by_sqrt_d:
        u = u * sqrt_d
    gate = gate * s_gate.view(1, 1, -1) * sqrt_d
    return u * F.silu(gate)


def scaled_logits_from_carried(
    y: Tensor,
    rho: Tensor,
    output_emb: Tensor,
    logit_scale: Tensor,
) -> Tensor:
    """Language-model logits from a carried hidden state.

    This computes the code-level enGPT output equation:

        logits[b,t,v] = <Y[b,t], E_out[v]> * s_z[v] / rho[b,t]

    The portable PyTorch path uses a standard GEMM-shaped expression because
    cuBLAS is faster than writing logits and then launching separate full-vocab
    row/column scaling passes. A fused CUDA/Triton CE or sampling kernel can
    replace this helper and consume the same row and column scales without
    writing full logits when the caller does not need them.
    """

    h = y / rho.unsqueeze(-1)
    return F.linear(h, output_emb * logit_scale.unsqueeze(-1))


def carried_residual(
    y: Tensor,
    rho: Tensor,
    branch: Tensor,
    alpha: Tensor,
    branch_eps: float,
    residual_eps: float,
) -> Tuple[Tensor, Tensor]:
    """Exact carried-radius replacement for nGPT branch and residual norms.

    The returned pair `(u, rho_plus)` represents the same hidden vector as:

        normalize((1 - alpha) * (y / rho) + alpha * normalize(branch))

    This implementation follows the single-reduction formula in
    `efficient-ngpt.md` and never materializes the normalized branch or
    normalized hidden tensor.
    """

    while alpha.ndim < branch.ndim:
        alpha = alpha.unsqueeze(0)
    beta = 1.0 - alpha
    rho_e = rho.unsqueeze(-1)

    branch_sq = branch.square()
    r0 = branch_sq.sum(dim=-1)
    r1 = (beta.square() * y.square()).sum(dim=-1)
    r2 = (alpha.square() * branch_sq).sum(dim=-1)
    r3 = (alpha * beta * y * branch).sum(dim=-1)

    cb = r0.clamp_min(0.0).sqrt()
    if branch_eps > 0:
        cb = cb.clamp_min(branch_eps)

    rho_cb = rho * cb
    u_norm_sq = cb.square() * r1 + rho.square() * r2 + 2.0 * rho_cb * r3
    u_norm = u_norm_sq.clamp_min(0.0).sqrt()
    rho_plus = u_norm
    if residual_eps > 0:
        rho_plus = torch.maximum(rho_plus, residual_eps * rho_cb)

    u = beta * cb.unsqueeze(-1) * y + alpha * rho_e * branch
    return u, rho_plus


def carried_residual_gauge(
    y: Tensor,
    rho: Tensor,
    branch: Tensor,
    alpha: Tensor,
    branch_eps: float,
    residual_eps: float,
    max_radius: float,
) -> Tuple[Tensor, Tensor]:
    """Carried residual with the exact gauge rescale folded into the same op.

    This is algebraically identical to:

        y, rho = carried_residual(...)
        y, rho = gauge_carried_state(y, rho, max_radius)

    Folding the gauge into the residual path avoids a separate full hidden
    transform in eager PyTorch and gives fused CUDA/Triton backends one logical
    primitive to replace.
    """

    while alpha.ndim < branch.ndim:
        alpha = alpha.unsqueeze(0)
    beta = 1.0 - alpha
    rho_e = rho.unsqueeze(-1)

    branch_sq = branch.square()
    r0 = branch_sq.sum(dim=-1)
    r1 = (beta.square() * y.square()).sum(dim=-1)
    r2 = (alpha.square() * branch_sq).sum(dim=-1)
    r3 = (alpha * beta * y * branch).sum(dim=-1)

    cb = r0.clamp_min(0.0).sqrt()
    if branch_eps > 0:
        cb = cb.clamp_min(branch_eps)

    rho_cb = rho * cb
    u_norm_sq = cb.square() * r1 + rho.square() * r2 + 2.0 * rho_cb * r3
    u_norm = u_norm_sq.clamp_min(0.0).sqrt()
    rho_plus = u_norm
    if residual_eps > 0:
        rho_plus = torch.maximum(rho_plus, residual_eps * rho_cb)

    u = beta * cb.unsqueeze(-1) * y + alpha * rho_e * branch
    if max_radius <= 0:
        return u, rho_plus
    gauge = (rho_plus / max_radius).clamp_min(1.0)
    return u / gauge.unsqueeze(-1), rho_plus / gauge


def gauge_carried_state(y: Tensor, rho: Tensor, max_radius: float) -> Tuple[Tensor, Tensor]:
    """Rescale a carried state without changing the represented hidden vector.

    `(y / scale) / (rho / scale) == y / rho`, so this is an exact gauge change.
    It prevents deep carried-radius stacks from overflowing raw Q/K projections
    before their radial scale cancellation is applied.
    """

    if max_radius <= 0:
        return y, rho
    scale = (rho / max_radius).clamp_min(1.0)
    return y / scale.unsqueeze(-1), rho / scale


def reference_residual(
    h: Tensor,
    branch: Tensor,
    alpha: Tensor,
    branch_eps: float,
    residual_eps: float,
) -> Tensor:
    while alpha.ndim < branch.ndim:
        alpha = alpha.unsqueeze(0)
    branch_hat = normalize_last_dim(branch, branch_eps)
    mixed = (1.0 - alpha) * h + alpha * branch_hat
    return normalize_last_dim(mixed, residual_eps)


def rotary_frequencies(
    seq_len: int,
    head_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    theta: float,
) -> Tuple[Tensor, Tensor]:
    rotary_dim = head_dim - (head_dim % 2)
    if rotary_dim == 0:
        empty = torch.empty(seq_len, 0, device=device, dtype=dtype)
        return empty, empty

    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to `[B, T, H, D]` tensors."""

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == 0:
        return x

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    ).flatten(-2)
    if x_pass.numel() == 0:
        return rotated
    return torch.cat((rotated, x_pass), dim=-1)


def apply_rope_inplace(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """In-place RoPE for inference/no-grad paths."""

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    x_even = x_rot[..., 0::2]
    x_odd = x_rot[..., 1::2]
    even = x_even.clone()
    odd = x_odd.clone()
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])
    x_even.copy_(even * cos - odd * sin)
    x_odd.copy_(even * sin + odd * cos)
    return x


def causal_sdpa(q: Tensor, k: Tensor, v: Tensor, *, scale: float, dropout_p: float) -> Tensor:
    """Causal SDPA for `[B, T, H, D]` inputs, returning `[B, T, H, D]`."""

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=True,
        scale=scale,
    )
    return out.transpose(1, 2)


def ngpt_attention_scale(head_dim: int) -> float:
    return math.sqrt(float(head_dim))


def gpt_attention_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(float(head_dim))
