# enGPT As Implemented

This note describes the code in this repository, not a separate model. The
source files are `engpt/config.py`, `engpt/kernels.py`, `engpt/models.py`, and
`engpt/optim.py`.

## Carried State

`EfficientNGPT.forward` carries a pair:

```python
y = self.token_emb(idx)
rho = torch.ones(idx.shape, device=idx.device, dtype=y.dtype)
```

The represented hidden vector is

```math
H_{b,t} = Y_{b,t} / \rho_{b,t}.
```

Each block receives `(Y, rho)` and returns the next `(Y, rho)`. The reference
path in `forward_reference` materializes `H` and is used by tests to check that
the carried path is the same computation.

## Attention

The block projects packed QKV from `Y`:

```python
qkv = self.qkv(y).view(bsz, seq_len, 3, n_head, head_dim)
q, k, v = qkv_postprocess_from_carried(qkv, rho, cos, sin, self.qk_scale, norm_eps)
```

`qkv_postprocess_from_carried` computes Q/K norms from raw projected `Y`.
For Q and K, the positive row scalar cancels under normalization:

```math
\frac{(Y W_q)/\rho}{\|(Y W_q)/\rho\|_2}
=
\frac{Y W_q}{\|Y W_q\|_2}.
```

The guarded denominator in code is therefore

```python
den = max(norm(raw), rho * norm_eps)
```

V is different because attention mixes values across tokens. The code divides
V by `rho`:

```python
v = v_raw / rho.view(bsz, seq_len, 1, 1)
```

The attention call then uses causal PyTorch SDPA with the nGPT scale:

```python
scale = sqrt(head_dim)
```

## Residual And Gauge

The carried residual represents

```math
normalize((1 - alpha) * (Y / rho) + alpha * normalize(branch)).
```

The implementation avoids writing the normalized branch and the normalized
post-residual hidden tensor. It computes

```math
c_b = max(||B||_2, eps_b)
```

```math
U_i = (1 - alpha_i)c_bY_i + alpha_i rho B_i
```

```math
rho_plus = max(||U||_2, eps_r rho c_b).
```

Then `U / rho_plus` is exactly the materialized residual result.

`carried_residual_gauge` folds in the gauge rescale:

```python
gauge = (rho_plus / max_radius).clamp_min(1.0)
return u / gauge.unsqueeze(-1), rho_plus / gauge
```

This preserves the represented hidden state because

```math
(U / g) / (rho_plus / g) = U / rho_plus.
```

The block calls `carried_residual_gauge` after attention and after the MLP.

## MLP

The portable PyTorch MLP helper currently materializes the represented hidden
state for the up/gate GEMM:

```python
mlp_in = y / rho.unsqueeze(-1)
u, gate = F.linear(mlp_in, weight).chunk(2, dim=-1)
```

It then applies the nGPT learned scales and SwiGLU:

```python
u = u * s_u
gate = gate * s_gate * sqrt(d_model)
hidden = u * silu(gate)
```

When `scale_mlp_u_by_sqrt_d` is enabled, `u` also receives the same
`sqrt(d_model)` factor.

## Output Head

The model delegates logits to `scaled_logits_from_carried`:

```python
logits = scaled_logits_from_carried(y, rho, output_emb, logit_scale)
```

The code-level equation is

```math
logits_{b,t,v}
=
<Y_{b,t}, E_v> * s_{z,v} / rho_{b,t}.
```

The current portable implementation uses the GEMM-shaped form:

```python
h = y / rho.unsqueeze(-1)
return F.linear(h, output_emb * logit_scale.unsqueeze(-1))
```

This is faster on CUDA than writing full logits and then launching separate
row-scale and column-scale passes. It is algebraically the same logit equation.

## Parameter Retraction And Gradient Projection

`project_parameters_` keeps nGPT weights on the intended unit spheres:

```python
normalize_rows_(token_emb.weight)
normalize_rows_(output_emb)
normalize_rows_(block.qkv.weight)
normalize_columns_(block.out_proj.weight)
normalize_rows_(block.up_gate.weight)
normalize_columns_(block.down_proj.weight)
```

`NGPTAdamW.step` projects gradients before AdamW and retracts parameters after
AdamW:

```python
project_ngpt_gradients_(model)
super().step()
project_ngpt_parameters_(model)
```

For row-normalized tensors,

```math
g <- g - <g, w>w.
```

For column-normalized tensors,

```math
g_j <- g_j - <g_j, w_j>w_j.
```

On CUDA, `engpt.kernels` uses optional Triton kernels for these row/column
normalization and gradient-projection operations when tensors are contiguous
2D CUDA tensors. If Triton is unavailable, or the tensor is not a supported CUDA
matrix, the same functions fall back to plain PyTorch. This keeps CPU, MPS,
ROCm-through-PyTorch, and non-Triton installs usable.

## Tests

The test suite checks:

- carried residual forward and gradient equality against the materialized
  reference residual;
- folded residual-plus-gauge equality against separate residual and gauge;
- full `EfficientNGPT.forward` equality against `forward_reference`;
- optimizer projection keeps constrained gradients tangent to the unit spheres;
- a tiny GPT and enGPT training step both run.
