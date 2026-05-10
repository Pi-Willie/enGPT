# enGPT From the Code

This note is derived only from the implementation files that program enGPT:

- `engpt/models.py`
- `engpt/kernels.py`
- `engpt/config.py`
- `engpt/optim.py`

It does not describe training results. It describes the model equations that the code executes.

## 1. The Carried State

`EfficientNGPT` does not keep the block hidden state only as a normalized vector `H`.
It carries two tensors:

```python
y = self.token_emb(idx)
rho = torch.ones(idx.shape, device=idx.device, dtype=y.dtype)
```

The represented hidden vector is

```math
H_{b,t} = \frac{Y_{b,t}}{\rho_{b,t}} .
```

Here `Y` has shape `[B, T, D]` and `rho` has shape `[B, T]`.
The radius is a positive per-token scalar. Dividing by it recovers the ordinary hidden vector.

Every block receives `(Y, rho)` and returns a new `(Y, rho)`:

```python
for block in self.blocks:
    y, rho = block(y, rho, cos, sin)
```

The purpose of carrying `rho` is simple: many nGPT normalizations are radial.
When a positive scalar multiplies a whole token row, the scalar often cancels in the next normalization.
The code keeps that scalar separate instead of repeatedly materializing normalized hidden vectors.

## 2. Parameters Kept on Spheres

The enGPT parameters are projected after initialization, and the optimizer projects them after every step.
The code uses row-normalized vectors for embeddings and input-side projections, and column-normalized vectors for output-side projections.

```python
normalize_rows_(self.token_emb.weight, eps)
normalize_rows_(self.output_emb, eps)

normalize_rows_(qkv[:d], eps)
normalize_rows_(qkv[d : 2 * d], eps)
normalize_rows_(qkv[2 * d :], eps)
normalize_columns_(self.out_proj.weight, eps)

normalize_rows_(self.up_gate.weight[:ff], eps)
normalize_rows_(self.up_gate.weight[ff:], eps)
normalize_columns_(self.down_proj.weight, eps)
```

The row projection is

```math
w_i \leftarrow \frac{w_i}{\max(\|w_i\|_2,\epsilon)} ,
```

and the column projection is

```math
W_{:,j} \leftarrow \frac{W_{:,j}}{\max(\|W_{:,j}\|_2,\epsilon)} .
```

During optimization, gradients are projected onto the tangent space of the same sphere constraints:

```python
grad.sub_((grad * weight).sum(dim=-1, keepdim=True) * weight)
grad.sub_(weight * (grad * weight).sum(dim=0, keepdim=True))
```

In equations, for row-normalized weights,

```math
g_i \leftarrow g_i - \langle g_i, w_i\rangle w_i ,
```

and for column-normalized weights,

```math
g_{:,j} \leftarrow g_{:,j} - \langle g_{:,j}, W_{:,j}\rangle W_{:,j}.
```

## 3. Attention Input From the Carried State

The block projects QKV from `Y`, not from `H`:

```python
qkv = self.qkv(y).view(bsz, seq_len, 3, self.cfg.n_head, self.cfg.head_dim)
q, k, v = qkv_postprocess_from_carried(
    qkv, rho, cos, sin, self.qk_scale, self.cfg.norm_eps
)
```

Let

```math
Q_{\text{raw}} = W_qY,\quad
K_{\text{raw}} = W_kY,\quad
V_{\text{raw}} = W_vY.
```

Because `Y = rho H` row-wise and the Q/K projections are bias-free,

```math
Q_{\text{raw}} = \rho W_qH,\quad
K_{\text{raw}} = \rho W_kH.
```

The code computes the Q/K denominators from the raw carried projections:

```python
q_den = torch.linalg.vector_norm(q_raw, ord=2, dim=-1)
k_den = torch.linalg.vector_norm(k_raw, ord=2, dim=-1)
guard = rho.unsqueeze(-1) * norm_eps
q_den = torch.maximum(q_den, guard)
k_den = torch.maximum(k_den, guard)
```

Thus

```math
\max(\|Q_{\text{raw}}\|_2,\rho\epsilon)
=
\rho\max(\|W_qH\|_2,\epsilon)
```

for positive `rho`, and similarly for keys.
RoPE preserves the L2 norm on the rotated dimensions, so it can be applied before the division without changing the denominator.

The code-level Q/K equations are

```math
Q =
\frac{\operatorname{RoPE}(Q_{\text{raw}})}
     {\max(\|Q_{\text{raw}}\|_2,\rho\epsilon)}
\odot S_{qk},
```

```math
K =
\frac{\operatorname{RoPE}(K_{\text{raw}})}
     {\max(\|K_{\text{raw}}\|_2,\rho\epsilon)}
\odot S_{qk}.
```

Values are different. Attention mixes values across tokens, so the row radius must be removed:

```python
v = v_raw / rho.view(bsz, seq_len, 1, 1)
```

which is

```math
V = W_vH.
```

Attention then uses PyTorch SDPA with the nGPT scale:

```python
attn = causal_sdpa(
    q,
    k,
    v,
    scale=ngpt_attention_scale(self.cfg.head_dim),
    dropout_p=0.0,
)
```

and

```math
\operatorname{ngpt\_attention\_scale}(d_h)=\sqrt{d_h}.
```

## 4. The Carried Residual

After attention or the MLP, the block has a branch tensor `B`.
The materialized reference operation is

```python
branch_hat = normalize_last_dim(branch, branch_eps)
mixed = (1.0 - alpha) * h + alpha * branch_hat
h_next = normalize_last_dim(mixed, residual_eps)
```

In equations,

```math
\hat B = \frac{B}{c_b},\qquad
c_b = \max(\|B\|_2,\epsilon_b),
```

```math
H^+ =
\frac{(1-\alpha)H + \alpha\hat B}
     {\max(\|(1-\alpha)H+\alpha\hat B\|_2,\epsilon_r)}.
```

The carried implementation represents the same `H+` without first writing `H = Y / rho`.
Multiplying the numerator by `rho * c_b` gives

```math
U = (1-\alpha)c_bY + \alpha\rho B.
```

The new carried radius is

```math
\rho^+ =
\max\left(\|U\|_2,\epsilon_r\rho c_b\right).
```

So the represented next hidden state is

```math
H^+ = \frac{U}{\rho^+}.
```

This is the function implemented by `carried_residual`.
The code computes `||U||` through scalar reductions:

```python
r1 = (beta.square() * y.square()).sum(dim=-1)
r2 = (alpha.square() * branch_sq).sum(dim=-1)
r3 = (alpha * beta * y * branch).sum(dim=-1)

u_norm_sq = cb.square() * r1 + rho.square() * r2 + 2.0 * rho_cb * r3
rho_plus = u_norm_sq.clamp_min(0.0).sqrt()
rho_plus = torch.maximum(rho_plus, residual_eps * rho_cb)

u = beta * cb.unsqueeze(-1) * y + alpha * rho_e * branch
```

`carried_residual_gauge` folds in the gauge rescale used by the model:

```python
gauge = (rho_plus / max_radius).clamp_min(1.0)
return u / gauge.unsqueeze(-1), rho_plus / gauge
```

This does not change the represented hidden vector:

```math
\frac{U/g}{\rho^+/g}=\frac{U}{\rho^+}.
```

It only keeps the carried radius bounded by `carried_gauge_max` when it grows too large.

## 5. MLP Path

The MLP path currently materializes `H = Y / rho` for the bias-free up/gate projection:

```python
mlp_in = y / rho.unsqueeze(-1)
u, gate = F.linear(mlp_in, weight).chunk(2, dim=-1)
```

Then it applies the learned nGPT channel scales:

```python
u = u * s_u.view(1, 1, -1)
if scale_u_by_sqrt_d:
    u = u * sqrt_d
gate = gate * s_gate.view(1, 1, -1) * sqrt_d
hidden = u * F.silu(gate)
```

In equations,

```math
U_m, G_m = W_{ug}H,
```

```math
U_m \leftarrow U_m \odot s_u \cdot
\begin{cases}
\sqrt D,&\text{if } \texttt{scale\_mlp\_u\_by\_sqrt\_d}\\
1,&\text{otherwise}
\end{cases}
```

```math
G_m \leftarrow G_m \odot s_g\sqrt D,
```

```math
\operatorname{MLP}(H)=W_d\left(U_m\odot\operatorname{silu}(G_m)\right).
```

The MLP branch then goes through the same carried residual-and-gauge operation as attention.

## 6. One Block

Putting the block code together:

```python
qkv = self.qkv(y).view(bsz, seq_len, 3, n_head, head_dim)
q, k, v = qkv_postprocess_from_carried(qkv, rho, cos, sin, self.qk_scale, norm_eps)

attn = causal_sdpa(q, k, v, scale=sqrt(head_dim), dropout_p=0.0)
branch = self.out_proj(attn.reshape(bsz, seq_len, -1))
y, rho = carried_residual_gauge(y, rho, branch, self.alpha_attn, ...)

hidden = carried_up_gate_swiglu(y, rho, self.up_gate.weight, self.s_u, self.s_gate, ...)
branch = self.down_proj(hidden)
y, rho = carried_residual_gauge(y, rho, branch, self.alpha_mlp, ...)
```

The block therefore has two carried residuals:

```math
(Y,\rho)
\xrightarrow{\text{attention branch}}
(Y_a,\rho_a)
\xrightarrow{\text{MLP branch}}
(Y_{\text{out}},\rho_{\text{out}}).
```

The trainable residual weights are exposed as positive values:

```python
alpha_attn = abs(alpha_attn_raw) * (alpha_init / scalar_init_scaling)
alpha_mlp = abs(alpha_mlp_raw) * (alpha_init / scalar_init_scaling)
```

with

```math
\texttt{scalar\_init\_scaling} = \frac{1}{\sqrt D}.
```

## 7. Output Logits

The public model returns logits.
The carried output helper preserves the carried-state logit equation:

```python
h = y / rho.unsqueeze(-1)
logits = F.linear(h, output_emb * logit_scale.unsqueeze(-1))
```

Equivalently,

```math
\operatorname{logits}_{b,t,v}
=
\frac{\langle Y_{b,t}, E_v\rangle}{\rho_{b,t}}
\cdot s_{z,v}.
```

Since `H = Y / rho`, this is the same as

```math
\operatorname{logits}_{b,t,v}
=
\langle H_{b,t}, E_v\rangle s_{z,v}.
```

The scale vector is

```python
logit_scale = self.s_z * (self.logit_scale_init_value / self.scalar_init_scaling)
```

and `s_z` is initialized at `1 / sqrt(D)`.

## 8. Reference Equality

The model keeps a materialized reference path in code:

```python
h = normalize_last_dim(self.token_emb(idx), self.cfg.norm_eps)
for block in self.blocks:
    h = block.forward_reference(h, cos, sin)
logits = F.linear(h, self.output_emb * self.logit_scale.unsqueeze(-1))
```

The tests compare the carried implementation against this reference:

```python
logits, loss = model(idx, targets)
ref_logits, ref_loss = model.forward_reference(idx, targets)

assert torch.allclose(logits, ref_logits, atol=1e-7, rtol=1e-6)
assert torch.allclose(loss, ref_loss, atol=1e-8, rtol=1e-7)
```

The carried state is therefore not a different model equation.
It is the same code-level nGPT computation represented as `(Y, rho)` so that radial factors can be cancelled or delayed where the implementation allows it.
