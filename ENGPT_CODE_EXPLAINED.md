# enGPT, As Implemented In Code

This note explains only the code in this repository. It does not rely on the
paper-style derivation in `efficient-ngpt.md`, except where the code itself
implements the same equations. The main implementation files are:

- `engpt/config.py`
- `engpt/kernels.py`
- `engpt/models.py`
- `engpt/optim.py`
- `tests/test_kernels.py`
- `tests/test_models.py`

The central class is `EfficientNGPT` in `engpt/models.py`. It is paired with
`GPTBaseline`, which exists only as a comparison model.

## 1. Model Configuration

`ModelConfig` defines the architectural and numerical constants:

```python
@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    alpha_init: float = 0.05
    logit_scale_init: float = 8.0
    scale_mlp_u_by_sqrt_d: bool = True
    carried_gauge_max: float = 1024.0
```

The derived dimensions are:

```python
head_dim = n_embd // n_head
mlp_width = int(mlp_ratio * n_embd)
```

The code requires `n_embd % n_head == 0`, positive model dimensions, positive
vocabulary size, and positive context length.

## 2. The Carried Hidden State

`EfficientNGPT.forward` does not store the hidden state only as a tensor `h`.
It stores a pair:

```python
y = self.token_emb(idx)
rho = torch.ones(idx.shape, device=idx.device, dtype=y.dtype)
```

The represented hidden state is:

```math
h = \frac{Y}{\rho}.
```

Here `Y` has shape `[B, T, d]` and `rho` has shape `[B, T]`. The scalar
`rho[b, t]` belongs to one token row and is broadcast over the embedding
dimension.

The main loop keeps passing this pair through every block:

```python
for block in self.blocks:
    y, rho = block(y, rho, cos, sin)
```

At the output head, the current PyTorch implementation materializes the
represented hidden state:

```python
h = y / rho.unsqueeze(-1)
logits = F.linear(h, self.output_emb * self.logit_scale.unsqueeze(-1))
```

So the code-level invariant is:

```math
\text{the represented hidden state after each block is } Y / \rho.
```

## 3. Parameter Initialization And Scalar Controls

The helper

```python
def _ngpt_base_scale(cfg: ModelConfig) -> float:
    return 1.0 / math.sqrt(float(cfg.n_embd))
```

sets the raw initialization scale for several learned scalar/vector controls.
Inside each `EfficientNGPTBlock`:

```python
base_scale = _ngpt_base_scale(cfg)
self.alpha_attn_raw = nn.Parameter(torch.full((cfg.n_embd,), base_scale))
self.alpha_mlp_raw = nn.Parameter(torch.full((cfg.n_embd,), base_scale))
self.s_qk = nn.Parameter(torch.full((cfg.n_head, cfg.head_dim), base_scale))
self.s_u = nn.Parameter(torch.ones(cfg.mlp_width))
self.s_gate = nn.Parameter(torch.ones(cfg.mlp_width))
```

The effective residual step vectors are nonnegative because the raw parameter is
taken through `abs()`:

```python
@property
def alpha_attn(self) -> Tensor:
    return self.alpha_attn_raw.abs() * (
        self.alpha_init_value / self.scalar_init_scaling
    )
```

The MLP residual step vector is analogous:

```python
@property
def alpha_mlp(self) -> Tensor:
    return self.alpha_mlp_raw.abs() * (
        self.alpha_init_value / self.scalar_init_scaling
    )
```

The Q/K scale used after Q/K normalization is:

```python
@property
def qk_scale(self) -> Tensor:
    return self.s_qk * (1.0 / self.scalar_init_scaling)
```

At model level, output logit scale is implemented as:

```python
self.s_z = nn.Parameter(torch.full((cfg.vocab_size,), self.scalar_init_scaling))

@property
def logit_scale(self) -> Tensor:
    return self.s_z * (
        self.logit_scale_init_value / self.scalar_init_scaling
    )
```

With the default config, the effective initial logit scale is `8.0`.

## 4. The Carried Residual Primitive

The most important low-level function is `carried_residual` in
`engpt/kernels.py`.

Its docstring states the function it represents:

```python
normalize((1 - alpha) * (y / rho) + alpha * normalize(branch))
```

The function returns a new pair `(u, rho_plus)` such that:

```math
\frac{u}{\rho^+}
=
\operatorname{normalize}
\left(
(1-\alpha)\frac{y}{\rho}
 + \alpha \operatorname{normalize}(b)
\right).
```

The code:

```python
def carried_residual(
    y: Tensor,
    rho: Tensor,
    branch: Tensor,
    alpha: Tensor,
    branch_eps: float,
    residual_eps: float,
) -> Tuple[Tensor, Tensor]:
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
```

The scalar `cb` is the guarded branch denominator:

```math
c_b = \max(\|b\|_2, \varepsilon_b).
```

The returned numerator is:

```math
u_i = (1-\alpha_i)c_b y_i + \alpha_i \rho b_i.
```

The returned denominator is:

```math
\rho^+
=
\max
\left(
\|u\|_2,
\varepsilon_r \rho c_b
\right).
```

This code path avoids writing the normalized branch tensor and the normalized
post-residual hidden tensor. The represented state is still recovered as
`u / rho_plus`.

The reference residual used for comparison is:

```python
def reference_residual(h, branch, alpha, branch_eps, residual_eps):
    branch_hat = normalize_last_dim(branch, branch_eps)
    mixed = (1.0 - alpha) * h + alpha * branch_hat
    return normalize_last_dim(mixed, residual_eps)
```

The test `test_carried_residual_matches_reference_hidden_and_gradients` checks
both forward equality and gradient equality against this reference form.

## 5. Exact Gauge Rescale

Deep carried states can have large raw `rho` even when the represented vector
`Y / rho` is normal. The code handles this with `gauge_carried_state`:

```python
def gauge_carried_state(y: Tensor, rho: Tensor, max_radius: float):
    if max_radius <= 0:
        return y, rho
    scale = (rho / max_radius).clamp_min(1.0)
    return y / scale.unsqueeze(-1), rho / scale
```

This preserves the represented hidden state exactly:

```math
\frac{Y / s}{\rho / s} = \frac{Y}{\rho}.
```

The code applies this after both carried residuals in each block:

```python
y, rho = gauge_carried_state(y, rho, self.cfg.carried_gauge_max)
```

The default cap is:

```python
carried_gauge_max: float = 1024.0
```

## 6. Attention Block

The attention path starts from carried state `(y, rho)`:

```python
qkv = self.qkv(y).view(bsz, seq_len, 3, self.cfg.n_head, self.cfg.head_dim)
q_raw, k_raw, v_raw = qkv.unbind(dim=2)
```

Note that Q, K, and V are projected from `y`, not from an already materialized
`h = y / rho`.

### 6.1 Q/K Denominators

The Q/K denominators are computed from the raw projected Q/K:

```python
q_den = torch.linalg.vector_norm(q_raw, ord=2, dim=-1)
k_den = torch.linalg.vector_norm(k_raw, ord=2, dim=-1)
```

With guarding enabled:

```python
guard = rho.unsqueeze(-1) * self.cfg.norm_eps
q_den = torch.maximum(q_den, guard)
k_den = torch.maximum(k_den, guard)
```

The guard includes `rho` because the represented query is based on `y / rho`.

### 6.2 RoPE, Q/K Normalization, And V Scaling

Training path:

```python
q = apply_rope(q_raw, cos, sin)
k = apply_rope(k_raw, cos, sin)
q = (q / q_den.unsqueeze(-1)) * scale
k = (k / k_den.unsqueeze(-1)) * scale
v = v_raw / rho.view(bsz, seq_len, 1, 1)
```

Inference/no-grad path mutates in place:

```python
q = apply_rope_inplace(q_raw, cos, sin)
k = apply_rope_inplace(k_raw, cos, sin)
q.div_(q_den.unsqueeze(-1)).mul_(scale)
k.div_(k_den.unsqueeze(-1)).mul_(scale)
v = v_raw.div_(rho.view(bsz, seq_len, 1, 1))
```

The represented query from hidden state `h = y / rho` would be:

```math
q_h = \frac{y W_Q}{\rho}.
```

After normalization, the positive row scalar cancels:

```math
\frac{q_h}{\|q_h\|_2}
=
\frac{(y W_Q)/\rho}{\|y W_Q\|_2/\rho}
=
\frac{y W_Q}{\|y W_Q\|_2}.
```

That is what the code computes from `q_raw`.

For V, the denominator does not cancel. The code explicitly computes:

```math
v = \frac{y W_V}{\rho}.
```

This is implemented by:

```python
v = v_raw / rho.view(bsz, seq_len, 1, 1)
```

### 6.3 Attention Scale

The attention call uses PyTorch SDPA:

```python
attn = causal_sdpa(
    q,
    k,
    v,
    scale=ngpt_attention_scale(self.cfg.head_dim),
    dropout_p=0.0,
)
```

The implemented nGPT attention scale is:

```python
def ngpt_attention_scale(head_dim: int) -> float:
    return math.sqrt(float(head_dim))
```

The GPT baseline uses:

```python
def gpt_attention_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(float(head_dim))
```

### 6.4 Attention Residual

After attention, the block projects the concatenated heads:

```python
branch = self.out_proj(attn.reshape(bsz, seq_len, -1))
```

Then it applies the carried residual:

```python
y, rho = carried_residual(
    y,
    rho,
    branch,
    self.alpha_attn,
    self.cfg.norm_eps,
    self.cfg.norm_eps,
)
y, rho = gauge_carried_state(y, rho, self.cfg.carried_gauge_max)
```

## 7. MLP Block

The current PyTorch MLP path materializes the represented hidden state:

```python
mlp_in = y / rho.unsqueeze(-1)
uv = self.up_gate(mlp_in)
u, gate = uv.chunk(2, dim=-1)
```

Then it applies learned MLP scales:

```python
u = u * self.s_u.view(1, 1, -1)
if self.cfg.scale_mlp_u_by_sqrt_d:
    u = u * math.sqrt(float(self.cfg.n_embd))
gate = gate * self.s_gate.view(1, 1, -1) * math.sqrt(float(self.cfg.n_embd))
```

The hidden activation is SwiGLU:

```python
hidden = u * F.silu(gate)
```

The MLP branch is:

```python
branch = self.down_proj(hidden)
```

Then the second carried residual is applied:

```python
y, rho = carried_residual(
    y,
    rho,
    branch,
    self.alpha_mlp,
    self.cfg.norm_eps,
    self.cfg.norm_eps,
)
y, rho = gauge_carried_state(y, rho, self.cfg.carried_gauge_max)
```

## 8. Materialized Reference Forward

`EfficientNGPT.forward_reference` is the explicit reference path used by tests
and benchmark exactness checks.

It starts with a materialized normalized embedding:

```python
h = normalize_last_dim(self.token_emb(idx), self.cfg.norm_eps)
```

Each block then runs `block.forward_reference(h, cos, sin)`.

Inside the reference block, Q and K are explicitly normalized:

```python
q = normalize_last_dim(apply_rope(q, cos, sin), self.cfg.norm_eps)
k = normalize_last_dim(apply_rope(k, cos, sin), self.cfg.norm_eps)
q = q * scale
k = k * scale
```

The reference residual explicitly normalizes the branch and then the mixed
hidden state:

```python
h = reference_residual(
    h,
    branch,
    self.alpha_attn,
    self.cfg.norm_eps,
    self.cfg.norm_eps,
)
```

The output head is the same as the carried path once `h` is materialized:

```python
logits = F.linear(h, self.output_emb * self.logit_scale.unsqueeze(-1))
```

The test `test_engpt_forward_matches_materialized_ngpt_reference` checks:

```python
assert torch.allclose(logits, ref_logits, atol=1e-7, rtol=1e-6)
assert torch.allclose(loss, ref_loss, atol=1e-8, rtol=1e-7)
```

## 9. Parameter Projection

`EfficientNGPT.project_parameters_` enforces unit-norm constraints on selected
parameter axes.

Embedding rows:

```python
normalize_rows_(self.token_emb.weight, eps)
normalize_rows_(self.output_emb, eps)
```

Packed Q/K/V rows:

```python
qkv = block.qkv.weight
d = self.cfg.n_embd
normalize_rows_(qkv[:d], eps)
normalize_rows_(qkv[d : 2 * d], eps)
normalize_rows_(qkv[2 * d :], eps)
```

Attention output projection columns:

```python
normalize_columns_(block.out_proj.weight, eps)
```

MLP up and gate rows:

```python
ff = self.cfg.mlp_width
normalize_rows_(block.up_gate.weight[:ff], eps)
normalize_rows_(block.up_gate.weight[ff:], eps)
```

MLP down projection columns:

```python
normalize_columns_(block.down_proj.weight, eps)
```

The report helper iterates over the same axes and returns the maximum unit-norm
error:

```python
def parameter_norm_report(self) -> Dict[str, float]:
    with torch.no_grad():
        max_err = 0.0
        for name, norms in self._normalized_parameter_norms():
            err = (norms - 1.0).abs().max().item()
            max_err = max(max_err, err)
        return {"max_unit_norm_error": max_err}
```

## 10. Optimizer Projection

`NGPTAdamW` subclasses `torch.optim.AdamW`.

Before the AdamW step, it projects gradients onto tangent spaces:

```python
project_ngpt_gradients_(self.model)
loss = super().step(closure=closure)
project_ngpt_parameters_(self.model)
```

For row-normalized tensors, the tangent projection is:

```python
grad.sub_((grad * weight).sum(dim=-1, keepdim=True) * weight)
```

Mathematically, for each row vector `w` and gradient `g`:

```math
g_\perp = g - \langle g, w \rangle w.
```

For column-normalized tensors, the projection is:

```python
grad.sub_(weight * (grad * weight).sum(dim=0, keepdim=True))
```

The test `test_ngpt_gradient_projection_is_tangent` checks that projected
gradients are orthogonal to the constrained axes:

```python
assert torch.allclose((q * qg).sum(dim=-1), torch.zeros(cfg.n_embd), atol=1e-5)
assert torch.allclose((out * outg).sum(dim=0), torch.zeros(cfg.n_embd), atol=1e-5)
```

## 11. Full Forward Summary

For input token IDs `idx`, `EfficientNGPT.forward` does:

```python
cos, sin = self._rope(idx.size(1), idx.device, self.token_emb.weight.dtype)
y = self.token_emb(idx)
rho = torch.ones(idx.shape, device=idx.device, dtype=y.dtype)

for block in self.blocks:
    y, rho = block(y, rho, cos, sin)

h = y / rho.unsqueeze(-1)
logits = F.linear(h, self.output_emb * self.logit_scale.unsqueeze(-1))
```

Each block does:

```text
1. q_raw, k_raw, v_raw = qkv(y)
2. q_den = ||q_raw||, k_den = ||k_raw||
3. q = RoPE(q_raw) / q_den * qk_scale
4. k = RoPE(k_raw) / k_den * qk_scale
5. v = v_raw / rho
6. attention = causal_sdpa(q, k, v, scale=sqrt(head_dim))
7. branch_attn = out_proj(attention)
8. (y, rho) = carried_residual(y, rho, branch_attn, alpha_attn)
9. (y, rho) = gauge_carried_state(y, rho)
10. mlp_in = y / rho
11. u, gate = up_gate(mlp_in)
12. hidden = scaled_u * silu(scaled_gate)
13. branch_mlp = down_proj(hidden)
14. (y, rho) = carried_residual(y, rho, branch_mlp, alpha_mlp)
15. (y, rho) = gauge_carried_state(y, rho)
```

## 12. What This File Claims, And What It Does Not Claim

This file claims only what the code implements:

- carried hidden states `(Y, rho)`;
- Q/K normalization from raw projected `Y`;
- explicit V division by `rho`;
- explicit MLP input materialization as `Y / rho` in the current PyTorch code;
- carried residuals equivalent to explicit branch-normalize plus residual-normalize;
- exact gauge rescale preserving `Y / rho`;
- row/column parameter retraction;
- tangent-gradient projection before AdamW steps;
- tests comparing the carried implementation to the materialized reference.

This file does not claim that every normalization has been removed. The code
still computes Q/K norms, still computes one carried residual reduction per
attention residual and one per MLP residual, and currently materializes `Y / rho`
before the MLP and output head in plain PyTorch.
