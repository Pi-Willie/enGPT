from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor


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
    x.div_(row_denominator(x, eps))
    return x


@torch.no_grad()
def normalize_columns_(x: Tensor, eps: float = 1e-12) -> Tensor:
    den = torch.linalg.vector_norm(x, ord=2, dim=0, keepdim=True).clamp_min(eps)
    x.div_(den)
    return x


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
