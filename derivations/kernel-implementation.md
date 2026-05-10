# Kernel Structure

This file is about implementation structure only. It does not report training
results.

enGPT is organized around a carried state `(Y, rho)` with represented hidden
state `H = Y / rho`. The kernels in `engpt/kernels.py` are the boundary where
the implementation avoids unnecessary materialization while staying equivalent
to the materialized nGPT reference path.

## Portable Kernel Boundaries

The current code has four main carried-state helpers.

### QKV Postprocess

```python
q, k, v = qkv_postprocess_from_carried(qkv, rho, cos, sin, qk_scale, norm_eps)
```

Input `qkv` is produced by the bias-free packed projection from `Y`.

The helper computes:

```math
Q = RoPE(Q_raw) / max(||Q_raw||_2, rho eps) * s_qk
```

```math
K = RoPE(K_raw) / max(||K_raw||_2, rho eps) * s_qk
```

```math
V = V_raw / rho.
```

The no-grad path applies RoPE and scaling in place. The grad path avoids
in-place mutation so autograd remains clean.

### Residual Plus Gauge

```python
y, rho = carried_residual_gauge(y, rho, branch, alpha, eps_b, eps_r, max_radius)
```

This folds the exact nGPT branch normalization, residual normalization, and
carried gauge rescale into one logical primitive:

```math
c_b = max(||B||_2, eps_b)
```

```math
U_i = (1 - alpha_i)c_bY_i + alpha_i rho B_i
```

```math
rho_plus = max(||U||_2, eps_r rho c_b)
```

```math
g = max(rho_plus / max_radius, 1).
```

It returns `(U / g, rho_plus / g)`, preserving `Y / rho` exactly.

### MLP Up/Gate

```python
hidden = carried_up_gate_swiglu(y, rho, weight, s_u, s_gate, ...)
```

The current portable helper materializes `Y / rho` before the up/gate GEMM, then
applies the nGPT MLP scales and SwiGLU. This is intentionally simple and
autograd-friendly. A future CUDA backend can replace the helper with a
row-scaled GEMM epilogue.

### Output Logits

```python
logits = scaled_logits_from_carried(y, rho, output_emb, logit_scale)
```

The equation is:

```math
logits_{b,t,v} = <Y_{b,t}, E_v> * s_{z,v} / rho_{b,t}.
```

The current CUDA-fast portable form materializes `H = Y / rho` and uses a
standard GEMM with a scaled output embedding. This is faster than separate
full-vocab row and column scaling passes unless a fused vocab cross-entropy or
sampling kernel is added.

## Optional Triton Retraction Kernels

The optimizer path has many small row/column projection operations. On CUDA,
`normalize_rows_`, `normalize_columns_`, `project_row_grad_`, and
`project_column_grad_` use Triton when the tensor is a contiguous 2D CUDA matrix.

The row gradient projection is:

```math
g_i <- g_i - <g_i, w_i>w_i.
```

The column gradient projection is:

```math
g_{:,j} <- g_{:,j} - <g_{:,j}, w_{:,j}>w_{:,j}.
```

The fallback is plain PyTorch, so importing the package does not require Triton.
This keeps the repo usable on CPU, Apple Silicon, and non-Triton PyTorch builds.

## Model Wiring

`EfficientNGPTBlock.forward` is intentionally small:

```python
qkv = self.qkv(y).view(bsz, seq_len, 3, n_head, head_dim)
q, k, v = qkv_postprocess_from_carried(...)
attn = causal_sdpa(q, k, v, scale=sqrt(head_dim), dropout_p=0.0)
branch = self.out_proj(attn.reshape(bsz, seq_len, -1))
y, rho = carried_residual_gauge(...)

hidden = carried_up_gate_swiglu(...)
branch = self.down_proj(hidden)
y, rho = carried_residual_gauge(...)
```

That layout keeps the algebraic boundaries explicit, so the Python fallback is
easy to audit and the CUDA/Triton replacement points are narrow.
