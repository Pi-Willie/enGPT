# enGPT: exact nGPT as a carried-radius transformer

## 0. Executive theorem

Let the reference model be nGPT as specified by the ICLR 2025 paper: embeddings, hidden states, attention/MLP matrices, branch outputs, and residual updates are normalized on the unit Euclidean hypersphere; residual updates use learned nonnegative eigen learning-rate vectors; Q/K are normalized and rescaled before attention; MLP intermediates and logits are rescaled; weight decay and warmup are removed. The paper’s summary explicitly says nGPT removes RMSNorm/LayerNorm, normalizes matrices and embeddings along the embedding dimension after each training step, replaces the usual residual equations by its normalized equations, changes attention scaling from (1/\sqrt{d_h}) to (\sqrt{d_h}), applies Q/K normalization/rescaling, rescales MLP intermediates, rescales logits, and removes weight decay/warmup. ([arXiv][1])

**Theorem.** enGPT is an implementation of that same nGPT function with no architectural approximation. In exact arithmetic, for every input sequence, every parameter value satisfying the same nGPT constraints, and every layer,

[
\operatorname{logits}_{\rm enGPT}(x)
====================================

\operatorname{logits}_{\rm nGPT}(x),
]

and the gradients with respect to all shared trainable parameters are identical.

enGPT’s exact forward pass uses, per block,

[
\boxed{2\text{ full }d_{\rm model}\text{-axis residual/branch reductions}}
]

plus

[
\boxed{2\text{ per-token, per-head }d_h\text{-axis Q/K norm reductions}}.
]

The two full hidden-width reductions are minimal for exact nGPT. The Q/K norm scalars are also irreducible for exact nGPT. They can be fused into QKV/attention kernels, but they cannot be deleted without changing the model. The nGPT paper notes that nGPT has six normalization steps per layer, two of them for (q) and (k), compared with two in GPT; it also reports significant step-time overhead and says the nGPT normalizations were not yet as optimized/fused as GPT normalizations. ([arXiv][1])

The key implementation move is not “eventually return to the sphere.” The key move is to store each hidden state as a **numerator plus a scalar denominator**:

[
(y,\rho),\qquad h=\frac{y}{\rho}.
]

Every operation that nGPT would apply to (h) is computed exactly from ((y,\rho)). Normalized hidden vectors need not be materialized at block boundaries or after attention. The current portable output helper uses a standard GEMM-shaped expression for throughput, while keeping the same carried-state logit equation.

---

# 1. Reference nGPT model

## 1.1 Normalization convention

For the unguarded mathematical model,

[
N(x)=\frac{x}{\lVert x\rVert_2},
\qquad x\ne0.
]

For production code, use the same guard as the reference implementation. Define

[
N_\varepsilon(x)=\frac{x}{c_\varepsilon(x)},
\qquad
c_\varepsilon(x)=\max(\lVert x\rVert_2,\varepsilon).
]

All exact equivalence statements below hold for (\varepsilon=0) on the nonzero domain, or for guarded nGPT when enGPT uses the same (\varepsilon) at the same logical normalization sites.

The nGPT paper states that its normalization function normalizes vectors to unit norm, without RMSNorm/LayerNorm-style learned elementwise scaling, and interprets this normalization as a retraction back to the hypersphere. ([arXiv][1])

## 1.2 Tensor shapes and right-multiply convention

Use row-token activations and right-multiply matrices:

[
xW.
]

Let

[
B=\text{batch size},\quad
T=\text{sequence length},\quad
d=d_{\rm model},\quad
H=\text{number of heads},\quad
d_h=d/H,\quad
d_{\rm ff}=\text{MLP width}.
]

Embeddings:

[
E_{\rm in}\in\mathbb R^{V\times d},
\qquad
E_{\rm out}\in\mathbb R^{V\times d}.
]

Per layer (\ell), per head (a),

[
W_Q^{\ell,a},W_K^{\ell,a},W_V^{\ell,a}\in\mathbb R^{d\times d_h},
]

or equivalently packed as (W_Q^\ell,W_K^\ell,W_V^\ell\in\mathbb R^{d\times d}).

Attention output:

[
W_O^\ell\in\mathbb R^{Hd_h\times d}.
]

MLP:

[
W_u^\ell,W_\nu^\ell\in\mathbb R^{d\times d_{\rm ff}},
\qquad
W_d^\ell\in\mathbb R^{d_{\rm ff}\times d}.
]

Learned forward scales:

[
s_{qk}^{\ell,a}\in\mathbb R^{d_h},
\qquad
s_u^\ell,s_\nu^\ell\in\mathbb R^{d_{\rm ff}},
\qquad
s_z\in\mathbb R^V.
]

Residual eigen learning-rate vectors:

[
\alpha_A^\ell,\alpha_M^\ell\in\mathbb R_{\ge0}^{d}.
]

The paper introduces Q/K scaling vectors, MLP intermediate scaling vectors, and logit scaling; it also describes the eigen learning-rate vectors used in the normalized residual equations. ([arXiv][1])

## 1.3 nGPT parameter axes

The phrase “normalize along the embedding dimension” is axis-dependent. In mathematical right-multiply notation:

[
\lVert E_{{\rm in},v:}\rVert_2
==============================

\lVert E_{{\rm out},v:}\rVert_2
=1.
]

For projections whose columns are embedding vectors compared with (h),

[
\lVert W_{Q,:j}^{\ell,a}\rVert_2
================================

# \lVert W_{K,:j}^{\ell,a}\rVert_2

\lVert W_{V,:j}^{\ell,a}\rVert_2
=1,
]

[
\lVert W_{u,:j}^{\ell}\rVert_2
==============================

\lVert W_{\nu,:j}^{\ell}\rVert_2
=1.
]

For output/down matrices, the embedding vectors are rows in the right-multiply convention:

[
\lVert W_{O,i:}^{\ell}\rVert_2=1,
\qquad
\lVert W_{d,i:}^{\ell}\rVert_2=1.
]

The official NVIDIA implementation’s post-step normalization is consistent with this: embeddings and the language-model head are normalized along their embedding row dimension; Q/K/V and MLP-up packed weights are normalized along the dimension corresponding to mathematical columns; attention output and MLP-down weights are normalized along the opposite dimension, corresponding to mathematical rows. ([GitHub][2])

---

# 2. enGPT state representation

nGPT stores hidden state (h_{\ell,t}\in\mathbb R^d) explicitly normalized.

enGPT stores

[
(y_{\ell,t},\rho_{\ell,t}),
\qquad
\rho_{\ell,t}>0,
\qquad
h_{\ell,t}^{\rm nGPT}=\frac{y_{\ell,t}}{\rho_{\ell,t}}.
]

For unguarded nGPT, (\rho_{\ell,t}=\lVert y_{\ell,t}\rVert_2). For guarded nGPT, (\rho_{\ell,t}) is the **normalization denominator**, not necessarily the Euclidean norm.

Initial embedding:

[
y_{0,t}=E_{{\rm in},x_t:},
\qquad
\rho_{0,t}=1.
]

Because (E_{\rm in}) rows are unit norm, (y_{0,t}/\rho_{0,t}) equals nGPT’s initial hidden vector.

---

# 3. The fused residual primitive

This is the core enGPT operation.

## 3.1 Reference residual operation

nGPT sublayers have the form

[
b=F(h),
]

[
\widehat b=N(b),
]

[
h^+=N!\left((1-\alpha)\odot h+\alpha\odot \widehat b\right).
]

For guarded nGPT,

[
\widehat b=N_{\varepsilon_b}(b)=\frac{b}{c_b},
\qquad
c_b=\max(\lVert b\rVert_2,\varepsilon_b),
]

[
h^+
===

N_{\varepsilon_r}
!\left((1-\alpha)\odot h+\alpha\odot \frac{b}{c_b}\right).
]

## 3.2 enGPT carried-denominator residual

Input:

[
h=\frac{y}{\rho},
\qquad
\rho>0,
\qquad
b\in\mathbb R^d.
]

Define

[
c_b=\max(\lVert b\rVert_2,\varepsilon_b).
]

Define the residual numerator

[
\boxed{
u_i
===

(1-\alpha_i)c_b,y_i
+
\alpha_i\rho,b_i.
}
]

Define the next denominator

[
\boxed{
\rho^+
======

\max\left(
\lVert u\rVert_2,,
\varepsilon_r,\rho,c_b
\right).
}
]

The enGPT residual primitive returns

[
\boxed{
\operatorname{FRes}_{\alpha}(y,\rho,b)
======================================

(u,\rho^+).
}
]

For unguarded nGPT, set (\varepsilon_b=\varepsilon_r=0), so

[
c_b=\lVert b\rVert_2,
\qquad
\rho^+=\lVert u\rVert_2.
]

## 3.3 Proof of exactness

Let

[
m
=

(1-\alpha)\odot \frac{y}{\rho}
+
\alpha\odot \frac{b}{c_b}.
]

This is exactly the vector that guarded nGPT normalizes in the residual update. Multiply by the positive scalar (\rho c_b):

[
\rho c_b,m
==========

(1-\alpha)\odot c_b y
+
\alpha\odot \rho b
==================

u.
]

Therefore

[
m=\frac{u}{\rho c_b}.
]

Guarded nGPT returns

[
N_{\varepsilon_r}(m)
====================

\frac{m}{\max(\lVert m\rVert_2,\varepsilon_r)}.
]

Since (\rho c_b>0),

[
\lVert m\rVert_2
================

\frac{\lVert u\rVert_2}{\rho c_b}.
]

Thus

[
N_{\varepsilon_r}(m)
====================

\frac{u/(\rho c_b)}
{\max(\lVert u\rVert_2/(\rho c_b),\varepsilon_r)}
=================================================

\frac{u}
{\max(\lVert u\rVert_2,\varepsilon_r\rho c_b)}
==============================================

\frac{u}{\rho^+}.
]

So the carried state ((u,\rho^+)) represents exactly the same hidden vector as nGPT:

[
\boxed{
\frac{u}{\rho^+}
================

h^+_{\rm nGPT}.
}
]

No normalized branch tensor (\widehat b) and no normalized hidden tensor (h^+) need to be materialized.

## 3.4 One-reduction formula

The primitive needs (c_b) and (\rho^+). Both come from one hidden-width reduction.

Let

[
\beta_i=1-\alpha_i.
]

Accumulate

[
R_0=\sum_i b_i^2,
]

[
R_1=\sum_i \beta_i^2y_i^2,
]

[
R_2=\sum_i \alpha_i^2b_i^2,
]

[
R_3=\sum_i \alpha_i\beta_i y_i b_i.
]

Then

[
c_b=\max(\sqrt{R_0},\varepsilon_b),
]

and

[
\lVert u\rVert_2^2
==================

c_b^2R_1+\rho^2R_2+2c_b\rho R_3.
]

Therefore

[
\rho^+
======

\max\left(
\sqrt{c_b^2R_1+\rho^2R_2+2c_b\rho R_3},
\varepsilon_r\rho c_b
\right).
]

After the reduction has produced (c_b) and (\rho^+), the kernel writes

[
u_i=\beta_i c_b y_i+\alpha_i\rho b_i.
]

This is the exact replacement for two explicit nGPT normalizations.

---

# 4. Exact attention from a carried state

Input to attention at token (t) is represented by

[
(y_t,\rho_t),
\qquad
h_t=\frac{y_t}{\rho_t}.
]

A tempting but wrong shortcut is to feed (y_t) directly into all of Q/K/V and rely on attention scale invariance. That fails because each token has its own (\rho_t). Q/K normalization cancels per-token radial scale in scores, but the value vectors would become (\rho_j v_j), so the attention output for token (t) would be

[
\sum_{j\le t}p_{tj}\rho_j v_j,
]

not

[
\sum_{j\le t}p_{tj}v_j.
]

enGPT fixes this by consuming the carried denominator in the V path, and in the MLP path later. The model does not assume attention is invariant to per-token radial scale.

## 4.1 Q/K/V projections

For head (a),

[
\widetilde q_{t,a}=y_tW_Q^{a},
]

[
\widetilde k_{t,a}=y_tW_K^{a},
]

[
v_{t,a}=\frac{y_tW_V^{a}}{\rho_t}.
]

The true nGPT query from (h_t) would be

[
q_{t,a}^{\rm nGPT}
==================

# \frac{y_tW_Q^a}{\rho_t}

\frac{\widetilde q_{t,a}}{\rho_t}.
]

Similarly,

[
k_{t,a}^{\rm nGPT}
==================

\frac{\widetilde k_{t,a}}{\rho_t}.
]

The V path is explicitly de-radialized, so

[
v_{t,a}
=======

# h_tW_V^a

v_{t,a}^{\rm nGPT}.
]

## 4.2 RoPE

Let (R_t) denote the RoPE rotation at position (t). RoPE is blockwise orthogonal; if a partial rotary dimension is used, the unrotated coordinates are multiplied by the identity, so the full operation is still orthogonal on the affected head vector.

Thus

[
\lVert R_t x\rVert_2=\lVert x\rVert_2.
]

So

[
N(R_t x)=R_tN(x)
]

for unguarded (N), and the same identity holds for (N_\varepsilon) because the denominator depends only on the Euclidean norm.

The nGPT paper applies RoPE to query and key and then normalizes/rescales (q) and (k); it also explains that Q/K normalization restores the per-head hypersphere after projections and RoPE. ([arXiv][1])

## 4.3 Exact Q/K normalization without materializing normalized Q/K

For unguarded nGPT, define

[
\widehat q_{t,a}
================

\frac{R_t\widetilde q_{t,a}}
{\lVert \widetilde q_{t,a}\rVert_2}
\odot s_{qk}^{a},
]

[
\widehat k_{t,a}
================

\frac{R_t\widetilde k_{t,a}}
{\lVert \widetilde k_{t,a}\rVert_2}
\odot s_{qk}^{a}.
]

This equals the explicit nGPT result because

[
\frac{R_t(\widetilde q_{t,a}/\rho_t)}
{\lVert \widetilde q_{t,a}/\rho_t\rVert_2}
==========================================

\frac{R_t\widetilde q_{t,a}}
{\lVert \widetilde q_{t,a}\rVert_2}.
]

For guarded nGPT with Q/K guard (\varepsilon_q), use

[
\widehat q_{t,a}
================

\frac{R_t\widetilde q_{t,a}}
{\max(\lVert \widetilde q_{t,a}\rVert_2,\rho_t\varepsilon_q)}
\odot s_{qk}^{a},
]

and similarly

[
\widehat k_{t,a}
================

\frac{R_t\widetilde k_{t,a}}
{\max(\lVert \widetilde k_{t,a}\rVert_2,\rho_t\varepsilon_k)}
\odot s_{qk}^{a}.
]

This is exactly

[
N_{\varepsilon_q}!\left(R_t\frac{\widetilde q_{t,a}}{\rho_t}\right)
\odot s_{qk}^a.
]

## 4.4 Attention scores

For token (t), key position (j), head (a),

[
S_{tja}
=======

\sqrt{d_h}
\sum_{m=1}^{d_h}
\widehat q_{t,a,m}\widehat k_{j,a,m}
+
M_{tj},
]

where

[
M_{tj}=
\begin{cases}
0, & j\le t,\
-\infty, & j>t.
\end{cases}
]

Equivalently, in raw-score form for unguarded nGPT,

[
S_{tja}
=======

\sqrt{d_h},
\frac{
\sum_m
(s_{qk,m}^{a})^2
(R_t\widetilde q_{t,a})*m
(R_j\widetilde k*{j,a})*m
}
{
\lVert \widetilde q*{t,a}\rVert_2
\lVert \widetilde k_{j,a}\rVert_2
}
+
M_{tj}.
]

The paper states that nGPT changes the attention scale from (1/\sqrt{d_h}) to (\sqrt{d_h}), because normalized query/key dot products have variance (1/d_h) and need (\sqrt{d_h}) to restore unit variance. ([arXiv][1])

Attention weights:

[
P_{tja}
=======

\frac{\exp(S_{tja})}
{\sum_{r\le t}\exp(S_{tra})}.
]

Head output:

[
c_{t,a}
=======

\sum_{j\le t}P_{tja}v_{j,a}.
]

Since (v_{j,a}=h_jW_V^a), this is exactly the nGPT value path.

Concatenate heads:

[
c_t=\operatorname{Concat}(c_{t,1},\dots,c_{t,H})\in\mathbb R^{Hd_h}.
]

Output projection:

[
b_{A,t}=c_tW_O^\ell.
]

Residual update in carried form:

[
(y_{A,t},\rho_{A,t})
====================

\operatorname{FRes}*{\alpha_A^\ell}(y*{\ell,t},\rho_{\ell,t},b_{A,t}).
]

Then

[
\frac{y_{A,t}}{\rho_{A,t}}
]

is exactly nGPT’s post-attention hidden state.

---

# 5. Exact MLP from a carried state

Input to the MLP is represented by

[
(y_{A,t},\rho_{A,t}),
\qquad
h_{A,t}=\frac{y_{A,t}}{\rho_{A,t}}.
]

Because SwiGLU is not scale-invariant, the denominator (\rho_{A,t}) must be consumed before the SiLU nonlinearity. enGPT does this in the MLP projection epilogue.

Raw projections:

[
u_t=\frac{y_{A,t}W_u^\ell}{\rho_{A,t}},
]

[
\nu_t=\frac{y_{A,t}W_\nu^\ell}{\rho_{A,t}}.
]

Scaled projections, using the paper’s default scaling convention:

[
\widetilde u_t=u_t\odot s_u^\ell,
]

[
\widetilde\nu_t=\nu_t\odot s_\nu^\ell\sqrt d.
]

The paper states that the (\sqrt{d_{\rm model}}) rescaling of the gate-side MLP intermediate is needed to benefit from SiLU’s nonlinearity. ([arXiv][1])

SwiGLU:

[
m_t=\widetilde u_t\odot\operatorname{SiLU}(\widetilde\nu_t),
]

where

[
\operatorname{SiLU}(x)=x\sigma(x).
]

Down projection:

[
b_{M,t}=m_tW_d^\ell.
]

Residual update in carried form:

[
(y_{\ell+1,t},\rho_{\ell+1,t})
==============================

\operatorname{FRes}*{\alpha_M^\ell}(y*{A,t},\rho_{A,t},b_{M,t}).
]

Then

[
\frac{y_{\ell+1,t}}{\rho_{\ell+1,t}}
]

is exactly nGPT’s post-MLP hidden state.

---

# 6. Output head, probabilities, and sampling

After (L) blocks, enGPT has

[
(y_{L,t},\rho_{L,t}),
\qquad
h_{L,t}=\frac{y_{L,t}}{\rho_{L,t}}.
]

nGPT logits:

[
z_{t,v}
=======

s_{z,v}
\left\langle
E_{{\rm out},v:},
h_{L,t}
\right\rangle.
]

enGPT computes

[
\boxed{
z_{t,v}
=======

s_{z,v}
\frac{
\left\langle E_{{\rm out},v:},y_{L,t}\right\rangle
}
{\rho_{L,t}}.
}
]

No final hidden normalization tensor is materialized. The row scalar (1/\rho_{L,t}) fuses into the output-head GEMM epilogue.

The nGPT paper introduces elementwise logit scaling because normalized embeddings make raw logits bounded dot products; it also says no additional normalization is required after the final layer. ([arXiv][1])

Training probability:

[
p_{t,v}
=======

\frac{\exp(z_{t,v})}{\sum_{r=1}^V\exp(z_{t,r})}.
]

Cross-entropy is identical because logits are identical.

Sampling is identical if it consumes the same logits with the same external sampling rule: same temperature, same top-(k), same top-(p), same repetition penalties if any, and same random bits. enGPT does not change sampling.

---

# 7. Full enGPT block specification

For one block (\ell), input is

[
Y_\ell\in\mathbb R^{B\times T\times d},
\qquad
\rho_\ell\in\mathbb R_{>0}^{B\times T}.
]

For each batch-token row (n=(b,t)), write (y_n=Y_{\ell,b,t,:}), (\rho_n=\rho_{\ell,b,t}).

## 7.1 Attention

For every head (a),

[
\widetilde q_{n,a}=y_nW_Q^{\ell,a},
]

[
\widetilde k_{n,a}=y_nW_K^{\ell,a},
]

[
v_{n,a}=(y_nW_V^{\ell,a})/\rho_n.
]

Apply RoPE:

[
q^{R}*{t,a}=R_t\widetilde q*{t,a},
\qquad
k^{R}*{t,a}=R_t\widetilde k*{t,a}.
]

Unguarded Q/K normalization:

[
\bar q_{t,a}
============

(q^{R}*{t,a}/\lVert\widetilde q*{t,a}\rVert_2)\odot s_{qk}^{\ell,a},
]

[
\bar k_{t,a}
============

(k^{R}*{t,a}/\lVert\widetilde k*{t,a}\rVert_2)\odot s_{qk}^{\ell,a}.
]

Guarded Q/K normalization replaces denominators by

[
\max(\lVert\widetilde q_{t,a}\rVert_2,\rho_t\varepsilon_q),
\qquad
\max(\lVert\widetilde k_{t,a}\rVert_2,\rho_t\varepsilon_k).
]

Scores:

[
S_{tja}
=======

\sqrt{d_h},\bar q_{t,a}^{\top}\bar k_{j,a}+M_{tj}.
]

Softmax:

[
P_{tja}=\operatorname{softmax}*{j\le t}(S*{tja}).
]

Head output:

[
c_{t,a}=\sum_{j\le t}P_{tja}v_{j,a}.
]

Concatenate:

[
c_t=[c_{t,1};\dots;c_{t,H}].
]

Output projection:

[
b_{A,t}=c_tW_O^\ell.
]

Carried residual:

[
(y_{A,t},\rho_{A,t})
====================

\operatorname{FRes}*{\alpha_A^\ell}
(y*{\ell,t},\rho_{\ell,t},b_{A,t}).
]

## 7.2 MLP

Projection from carried hidden:

[
u_t=(y_{A,t}W_u^\ell)/\rho_{A,t},
]

[
\nu_t=(y_{A,t}W_\nu^\ell)/\rho_{A,t}.
]

Scale:

[
\widetilde u_t=u_t\odot s_u^\ell,
]

[
\widetilde\nu_t=\nu_t\odot s_\nu^\ell\sqrt d.
]

Activation:

[
m_t=\widetilde u_t\odot\operatorname{SiLU}(\widetilde\nu_t).
]

Down projection:

[
b_{M,t}=m_tW_d^\ell.
]

Carried residual:

[
(y_{\ell+1,t},\rho_{\ell+1,t})
==============================

\operatorname{FRes}*{\alpha_M^\ell}
(y*{A,t},\rho_{A,t},b_{M,t}).
]

Return

[
(Y_{\ell+1},\rho_{\ell+1}).
]

---

# 8. Exact-equivalence proof

## Lemma 1: carried representation is exact at initialization

Because every input embedding row is normalized,

[
\lVert E_{{\rm in},x_t:}\rVert_2=1.
]

enGPT initializes

[
y_{0,t}=E_{{\rm in},x_t:},
\qquad
\rho_{0,t}=1.
]

Therefore

[
y_{0,t}/\rho_{0,t}=E_{{\rm in},x_t:}=h_{0,t}^{\rm nGPT}.
]

## Lemma 2: Q/K normalization from (y) equals Q/K normalization from (h=y/\rho)

For unguarded nGPT,

[
q_h=\frac{yW_Q}{\rho}.
]

Since (\rho>0),

[
N(q_h)
======

# \frac{yW_Q/\rho}{\lVert yW_Q/\rho\rVert_2}

\frac{yW_Q}{\lVert yW_Q\rVert_2}.
]

RoPE is orthogonal, so

[
N(Rq_h)=RN(q_h)
===============

\frac{R(yW_Q)}{\lVert yW_Q\rVert_2}.
]

The same proof applies to (k). Thus enGPT’s Q/K score vectors equal nGPT’s.

For guarded nGPT,

[
N_{\varepsilon_q}(Rq_h)
=======================

\frac{R(yW_Q)/\rho}
{\max(\lVert yW_Q\rVert_2/\rho,\varepsilon_q)}
==============================================

\frac{R(yW_Q)}
{\max(\lVert yW_Q\rVert_2,\rho\varepsilon_q)}.
]

This is exactly enGPT’s guarded denominator.

## Lemma 3: V projection from carried state equals nGPT’s V projection

nGPT uses

[
v_h=hW_V.
]

enGPT uses

[
v_y=\frac{yW_V}{\rho}.
]

Since (h=y/\rho),

[
v_y=(y/\rho)W_V=hW_V=v_h.
]

## Lemma 4: enGPT attention output equals nGPT attention output

By Lemma 2, all Q/K score vectors are identical to nGPT’s, hence all attention scores and softmax probabilities are identical. By Lemma 3, all values are identical. Therefore every head output (c_{t,a}), the concatenated vector (c_t), and the output projection (b_{A,t}=c_tW_O) are identical to nGPT.

## Lemma 5: (\operatorname{FRes}) equals nGPT’s branch-normalize-plus-residual-normalize update

This was proven in Section 3.3. For each token row,

[
\frac{y_A}{\rho_A}
==================

N_{\varepsilon_r}!\left(
(1-\alpha_A)\odot h
+
\alpha_A\odot N_{\varepsilon_b}(b_A)
\right).
]

Thus enGPT’s carried post-attention state represents exactly nGPT’s post-attention hidden vector.

## Lemma 6: MLP projections from carried state equal nGPT’s MLP projections

Given

[
h_A=\frac{y_A}{\rho_A},
]

enGPT computes

[
u=\frac{y_AW_u}{\rho_A}
=======================

h_AW_u,
]

[
\nu=\frac{y_AW_\nu}{\rho_A}
===========================

h_AW_\nu.
]

Therefore the scaled MLP intermediates, SwiGLU output, and down projection (b_M) are identical to nGPT’s.

## Lemma 7: enGPT’s MLP residual state represents nGPT’s post-MLP hidden state

Apply Lemma 5 with (b=b_M) and (\alpha=\alpha_M). Then

[
\frac{y_{\ell+1}}{\rho_{\ell+1}}
================================

N_{\varepsilon_r}!\left(
(1-\alpha_M)\odot h_A
+
\alpha_M\odot N_{\varepsilon_b}(b_M)
\right),
]

which is exactly nGPT’s post-MLP hidden state.

## Theorem 1: forward equivalence

Induct over layers.

Base case follows from Lemma 1.

Assume

[
y_{\ell,t}/\rho_{\ell,t}=h_{\ell,t}^{\rm nGPT}
]

for every token. Lemmas 2–4 show the attention branch raw output is identical to nGPT’s. Lemma 5 shows the carried post-attention state represents exactly nGPT’s post-attention hidden state. Lemma 6 shows the MLP branch is identical. Lemma 7 shows the carried post-MLP state represents exactly nGPT’s next hidden state.

Therefore the invariant holds for (\ell+1). By induction it holds for all layers.

The output head computes

[
s_{z,v}\langle E_{{\rm out},v:},y_L/\rho_L\rangle,
]

which is exactly nGPT’s logit. Hence

[
\operatorname{logits}_{\rm enGPT}(x)
====================================

\operatorname{logits}_{\rm nGPT}(x).
]

## Theorem 2: backward equivalence

Away from guard kinks, every enGPT fused primitive is an algebraic rewriting of the same differentiable function computed by nGPT. The composed map from parameters to logits is therefore the same differentiable map. Reverse-mode differentiation computes the adjoint of that same map, so parameter gradients are identical.

At guard kinks (\lVert x\rVert=\varepsilon), the guarded normalization itself is nondifferentiable. Exact gradient identity requires enGPT to use the same subgradient convention as the reference implementation. For unguarded nGPT, zero-norm normalization arguments are outside the mathematical domain; if the reference returns NaN, enGPT should return the same undefined/NaN behavior rather than silently substituting a different vector.

---

# 9. Minimality of exact enGPT

## 9.1 A coordinatewise map cannot normalize a vector

For (d\ge2), no coordinatewise function

[
f_i(x)=\phi_i(x_i)
]

can equal

[
N(x)_i=\frac{x_i}{\lVert x\rVert_2}
]

for all nonzero (x).

Proof. In (d=2), take

[
x=(1,1),
\qquad
y=(1,2).
]

Both have first coordinate (1), so a coordinatewise map gives

[
f_1(x)=f_1(y).
]

But

[
N(x)_1=\frac{1}{\sqrt2},
\qquad
N(y)_1=\frac{1}{\sqrt5}.
]

Contradiction. The (d>2) case contains this two-coordinate subspace. ∎

## 9.2 Each nGPT residual sublayer requires hidden-width cross-coordinate information

The exact residual map is

[
h^+
===

N!\left((1-\alpha)\odot h+\alpha\odot N(b)\right).
]

If at least one (\alpha_i\ne0), the output depends on (N(b)). By Lemma 9.1, (N(b)) cannot be determined coordinatewise for all (b). Therefore any exact implementation must compute cross-coordinate information about (b), at least enough to determine its normalization denominator or an equivalent scalar.

Thus each residual sublayer requires at least one hidden-width reduction-class operation.

## 9.3 Two full hidden-width reductions per block are necessary

Each nGPT block has two residual sublayers: attention and MLP. By Section 9.2, each requires one hidden-width cross-coordinate reduction. Therefore exact nGPT requires at least

[
\boxed{2}
]

full (d)-axis reductions per block.

enGPT attains this bound: one (\operatorname{FRes}) reduction after attention output projection and one (\operatorname{FRes}) reduction after MLP down projection.

## 9.4 Exact Q/K normalization is irreducible

For one head, exact nGPT scores have the form

[
S(q,k)
======

\sqrt{d_h}
\frac{q^\top D_s k}{\lVert q\rVert_2\lVert k\rVert_2},
]

where

[
D_s=\operatorname{Diag}(s_{qk}^2).
]

This score is invariant under independent positive rescaling:

[
S(\lambda q,k)=S(q,k),
\qquad
S(q,\mu k)=S(q,k),
\qquad
\lambda,\mu>0.
]

A raw dot score

[
\widetilde S(q,k)=\sqrt{d_h},q^\top D_s k
]

is not invariant:

[
\widetilde S(\lambda q,k)=\lambda\widetilde S(q,k).
]

Thus an exact implementation must know (\lVert q\rVert_2) and (\lVert k\rVert_2), or equivalent scalar information.

The weight constraints do not make Q/K norms constant. If (q=hA) had constant norm (c) for every unit (h), then

[
\lVert hA\rVert_2^2=hAA^\top h^\top=c^2
]

for every unit (h), implying

[
AA^\top=c^2I_d.
]

But

[
\operatorname{rank}(AA^\top)\le d_h.
]

For a transformer head (d_h<d), this is impossible unless (c=0), which would make the query identically zero. The same argument applies to keys.

Therefore Q/K norm scalars are irreducible for exact nGPT. They may be fused into QKV projection or attention score kernels; they may not be removed.

---

# 10. GPU-native kernel sequence

This section describes the exact enGPT forward pass in implementation terms. It is not CPU code and does not require off-GPU normalization.

## 10.1 Stored tensors

Persistent per-layer activations:

[
Y\in\mathbb R^{B\times T\times d},
\qquad
\rho\in\mathbb R^{B\times T}.
]

The normalized hidden tensor (H=Y/\rho) is not stored.

## 10.2 Per-block forward kernels

### Kernel group 1: QKV projection with carried-denominator handling

Inputs from HBM:

[
Y,\rho,W_Q,W_K,W_V.
]

Outputs:

[
\widetilde Q=YW_Q,
\qquad
\widetilde K=YW_K,
\qquad
V=(YW_V)\oslash\rho.
]

Here (\oslash\rho) means rowwise division by (\rho_{b,t}).

Fusion:

* (1/\rho) is a row scalar in the V epilogue.
* Q and K may be left radially unscaled because their later normalization cancels (\rho).
* If the same packed QKV GEMM is used, the V slice epilogue applies (1/\rho); Q/K slices do not need it.

HBM:

* Writes raw (\widetilde Q,\widetilde K) or passes them directly to the attention kernel depending on implementation.
* Writes exact (V) or stores it in KV cache during inference.

### Kernel group 2: Q/K norm accumulation and RoPE

For each ((B,T,H)) row, accumulate

[
r_Q=\lVert\widetilde q\rVert_2,
\qquad
r_K=\lVert\widetilde k\rVert_2.
]

Apply RoPE, which does not change those norms.

Fusion:

* Can be fused into the QKV projection epilogue if the projection kernel owns each head slice.
* Can be fused into the attention score kernel by accumulating the norm before using the row in dot products.
* In inference, normalized/scaled K may be stored in the KV cache to avoid recomputing key norms.

HBM:

* Normalized Q/K tensors do not need to be written.
* Store only reciprocal denominators or directly apply them inside the score calculation.

### Kernel group 3: FlashAttention-style score, softmax, and value accumulation

Compute

[
S_{tja}
=======

\sqrt{d_h}
\frac{
(q^R_{t,a})^\top D_s(k^R_{j,a})
}
{
r_{Q,t,a}r_{K,j,a}
}
+
M_{tj}
]

or the guarded denominator version.

Then compute

[
c_{t,a}=\sum_{j\le t}\operatorname{softmax}(S_{t,a})*j V*{j,a}.
]

Fusion:

* This is a FlashAttention-compatible modification: the score path multiplies by stored or computed reciprocal Q/K norms and by (s_{qk}).
* No normalized Q/K tensor needs to touch HBM.

### Kernel group 4: attention output projection plus carried residual

Compute

[
b_A=cW_O.
]

Then run (\operatorname{FRes}_{\alpha_A}(Y,\rho,b_A)), producing

[
Y_A,\rho_A.
]

Preferred fused implementation:

* Projection kernel computes (b_A) tiles and accumulates the four row scalars

[
R_0,R_1,R_2,R_3
]

needed by Section 3.4.

* After row scalars are finalized, the epilogue writes (Y_A) and (\rho_A), not normalized (h_A), not (N(b_A)).

Fallback exact implementation:

* Write (b_A) once.
* A single fused residual-reduction kernel reads (Y,\rho,b_A,\alpha_A), computes (Y_A,\rho_A), and discards (b_A).
* This fallback is still exact and still halves the explicit nGPT hidden normalization chain, but the preferred custom epilogue avoids the intermediate branch-output HBM roundtrip.

### Kernel group 5: MLP up/gate projection with carried-denominator handling

Inputs:

[
Y_A,\rho_A,W_u,W_\nu.
]

Compute

[
u=(Y_AW_u)\oslash\rho_A,
]

[
\nu=(Y_AW_\nu)\oslash\rho_A.
]

Apply forward scales:

[
\widetilde u=u\odot s_u,
]

[
\widetilde\nu=\nu\odot s_\nu\sqrt d.
]

Fusion:

* (1/\rho_A), (s_u), and (s_\nu\sqrt d) are epilogue row/column scales.
* No normalized hidden (Y_A/\rho_A) is materialized.

### Kernel group 6: SwiGLU activation

Compute

[
m=\widetilde u\odot\operatorname{SiLU}(\widetilde\nu).
]

This is an elementwise kernel unless fused with the MLP projection epilogue or the down-projection input transform.

### Kernel group 7: MLP down projection plus carried residual

Compute

[
b_M=mW_d.
]

Then run

[
(Y_{\ell+1},\rho_{\ell+1})
==========================

\operatorname{FRes}_{\alpha_M}(Y_A,\rho_A,b_M).
]

Preferred fusion is the same as attention output projection: accumulate (R_0,R_1,R_2,R_3) in the down-projection epilogue and write only (Y_{\ell+1},\rho_{\ell+1}).

### Output head kernel

For training all positions or inference last position, compute

[
z=s_z\odot\left((YE_{\rm out}^\top)\oslash\rho\right).
]

Fusion:

* (1/\rho) is a row epilogue scale.
* (s_z) is a column epilogue scale.
* No final normalized hidden state is stored.

---

# 11. Inference KV cache

During autoregressive inference, for each new token:

1. Store the carried state ((y_t,\rho_t)) only as needed for the current block computation.
2. For each layer/head, compute and cache the exact nGPT key representation used by attention:

[
\bar k_{t,a}
============

\left(
\frac{R_t(y_tW_K^a)}
{\lVert y_tW_K^a\rVert_2}
\right)
\odot s_{qk}^a
]

or the guarded version.

3. Cache the exact nGPT value

[
v_{t,a}=\frac{y_tW_V^a}{\rho_t}.
]

Then future tokens attend to cached (\bar k) and (v) exactly as nGPT would.

---

# 12. Corrected optimizer on the nGPT manifold

The optimizer is not part of the forward equivalence theorem, but it must preserve nGPT’s parameter constraints exactly. The nGPT paper says matrices and embeddings are normalized after each training step along their embedding dimension, and it removes weight decay and warmup. ([arXiv][1])

The default optimizer for this implementation is **spherical AdamW**:

* It operates on the correct nGPT normalized vector axes.
* It projects gradients onto the tangent space of the product of spheres.
* It applies AdamW moments to the projected gradients.
* It uses no weight decay on normalized vectors.
* It retracts every normalized vector exactly back to unit norm.
* It leaves unconstrained scalar/vector parameters on ordinary AdamW with zero weight decay.

This is the optimizer implemented as `NGPTAdamW` in `engpt/optim.py` and built by `build_ngpt_adamw`. It is deliberately conservative: it changes the vanilla AdamW step only where the nGPT manifold requires it.

## 12.1 Manifold

For each normalized matrix, orient its normalized vectors as rows.

Define an orientation operator (\Omega_W):

* If (W) is row-normalized in mathematical notation, (\Omega_W(W)=W).
* If (W) is column-normalized in mathematical notation, (\Omega_W(W)=W^\top).

Let

[
A=\Omega_W(W)\in\mathbb R^{p\times q}.
]

The constraint is

[
\mathcal M_A
============

{A\in\mathbb R^{p\times q}:\lVert A_{i:}\rVert_2=1\ \forall i}.
]

This is a product of (p) unit spheres.

## 12.2 Tangent space

At (A),

[
T_A\mathcal M_A
===============

{Z\in\mathbb R^{p\times q}:
\langle Z_{i:},A_{i:}\rangle=0\ \forall i}.
]

The orthogonal tangent projection is rowwise:

[
\boxed{
P_A(G)_{i:}
===========

## G_{i:}

\langle G_{i:},A_{i:}\rangle A_{i:}.
}
]

### Proof that (P_A) is tangent and orthogonal

For each row,

[
\langle P_A(G)*{i:},A*{i:}\rangle
=================================

## \langle G_{i:},A_{i:}\rangle

\langle G_{i:},A_{i:}\rangle
\lVert A_{i:}\rVert_2^2.
]

Because (\lVert A_{i:}\rVert_2=1),

[
\langle P_A(G)*{i:},A*{i:}\rangle=0.
]

So (P_A(G)\in T_A\mathcal M_A).

For any tangent (Z),

[
\langle G-P_A(G),Z\rangle_F
===========================

\sum_i
\langle
\langle G_{i:},A_{i:}\rangle A_{i:},
Z_{i:}
\rangle
=======

\sum_i
\langle G_{i:},A_{i:}\rangle
\langle A_{i:},Z_{i:}\rangle
============================

0.

]

Thus (P_A) is the orthogonal projection onto the tangent space.

## 12.3 Retraction

Given tangent update (D), define

[
\boxed{
R_A(-\eta D)_{i:}
=================

\frac{A_{i:}-\eta D_{i:}}
{\lVert A_{i:}-\eta D_{i:}\rVert_2}.
}
]

### Proof of constraint preservation

By construction,

[
\lVert R_A(-\eta D)_{i:}\rVert_2
================================

\frac{\lVert A_{i:}-\eta D_{i:}\rVert_2}
{\lVert A_{i:}-\eta D_{i:}\rVert_2}
===================================

1,
]

provided the denominator is nonzero. If (D_{i:}\perp A_{i:}), then (A_{i:}-\eta D_{i:}\ne0) for all finite (\eta), because its squared norm is

[
1+\eta^2\lVert D_{i:}\rVert_2^2>0.
]

Thus retraction preserves the nGPT constraint exactly.

## 12.4 Spherical AdamW update

Orient all quantities as rows:

[
A=\Omega_W(W),
\qquad
G_A=\Omega_W(G_W).
]

Compute current tangent gradient:

[
T=P_A(G_A).
]

Spherical AdamW first replaces the raw Euclidean gradient by (T). AdamW then updates its first and second moment estimates from this tangent gradient:

[
m_t=\beta_1m_{t-1}+(1-\beta_1)T,
]

[
v_t=\beta_2v_{t-1}+(1-\beta_2)T\odot T.
]

With bias correction, the preconditioned update is

[
D_t=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
]

The optimizer applies this update in the ambient coordinates and immediately retracts:

[
A^+=R_A(-\eta D_t).
]

For column-normalized parameters, the same operation is applied to (A=\Omega_W(W)=W^\top), then transposed back. For row-normalized parameters, it is applied directly to (W).

The AdamW implementation has zero weight decay for enGPT parameters. The "W" in the implementation name reflects use of PyTorch's AdamW machinery and decoupled-weight-decay interface, not a nonzero decay term on constrained vectors.

## 12.5 Why projection before AdamW is required

If the raw gradient has a radial component, an ambient AdamW update can spend optimizer state on directions that the retraction immediately removes. Worse, the moment buffers can keep accumulating radial components that do not represent motion on the product of spheres.

Projecting first makes every stored Adam moment depend only on feasible first-order motion:

[
T\in T_A\mathcal M_A.
]

The subsequent elementwise Adam preconditioning is not itself a Riemannian natural gradient, but the final retraction restores the exact nGPT constraint after every optimizer step. This gives a simple, stable, and implemented default optimizer whose behavior is easy to audit.

## 12.6 Final optimizer stack

Use these parameter groups:

[
E_{\rm in},E_{\rm out}:
\quad
\text{row-spherical AdamW with tangent-gradient projection and exact retraction}.
]

[
W_Q,W_K,W_V:
\quad
\text{row-spherical AdamW on mathematical rows}.
]

[
W_O:
\quad
\text{column-spherical AdamW on mathematical columns}.
]

[
W_u,W_\nu:
\quad
\text{row-spherical AdamW on mathematical rows}.
]

[
W_d:
\quad
\text{column-spherical AdamW on mathematical columns}.
]

[
\alpha_A,\alpha_M,s_{qk},s_u,s_\nu,s_z:
\quad
\text{AdamW with zero weight decay}.
]

No weight decay is applied to normalized vectors. After every optimizer step, every normalized vector is retracted to unit norm exactly.

---

# 13. Approximation section

## 13.1 enGPT accepts no architectural approximations

The exact enGPT architecture above has no deviation from nGPT in exact arithmetic. Its approximation list is empty:

[
\boxed{
X=\varnothing,\qquad Y=0.
}
]

All hidden-state normalizations are algebraically represented by carried numerator/denominator pairs. All Q/K normalizations are still computed exactly, but fused into the QKV/attention path. The model does not remove Q/K normalization.

## 13.2 Finite-precision caveat

Exact real-arithmetic equivalence does not imply bitwise equality to a particular unfused PyTorch nGPT implementation under bf16/fp16/fp32. Fusing reductions changes rounding order. To get bitwise equality, the reference nGPT and enGPT must define the same reduction tree, the same guard constants, the same cast points, and the same subgradient behavior at guard kinks. Without that, enGPT is mathematically identical but not guaranteed bit-identical in floating-point execution.

## 13.3 The closest vanilla-cost ablation: remove Q/K norm

This is not exact enGPT. It is the nGPT “without QK norm” ablation. The paper reports that removing Q/K normalization reduces training time in an ablation but worsens context-length extrapolation; it also explains that Q/K normalization restores per-head query/key vectors to a hypersphere after projections and RoPE. ([arXiv][1])

For one head, exact score:

[
S(q,k)
======

\sqrt{d_h}
\frac{q^\top D_s k}
{\lVert q\rVert_2\lVert k\rVert_2}.
]

No-QK score:

[
\widetilde S(q,k)
=================

\sqrt{d_h},q^\top D_s k.
]

Let

[
r=\lVert q\rVert_2,
\qquad
\kappa=\lVert k\rVert_2,
\qquad
s_\infty=\lVert s_{qk}\rVert_\infty.
]

Then

[
\widetilde S-S
==============

\sqrt{d_h},
q^\top D_s k
\left(
1-\frac{1}{r\kappa}
\right).
]

By Cauchy–Schwarz,

[
|q^\top D_s k|
\le
\lVert D_s\rVert_2\lVert q\rVert_2\lVert k\rVert_2
==================================================

s_\infty^2r\kappa.
]

Therefore

[
\boxed{
|\widetilde S-S|
\le
\sqrt{d_h},s_\infty^2|r\kappa-1|.
}
]

This bound is sharp for fixed (r,\kappa): choose (q) and (k) collinear in a coordinate attaining (s_\infty).

Under only nGPT’s unit-hidden and unit-column weight constraints, every query coordinate is a dot product of two unit vectors, so (|q_m|\le1). Hence

[
r\le\sqrt{d_h},
\qquad
\kappa\le\sqrt{d_h}.
]

Thus

[
r\kappa\in[0,d_h].
]

The sharp unconditional uniform score bound is

[
\boxed{
|\widetilde S-S|
\le
\sqrt{d_h},s_\infty^2
\max(1,d_h-1).
}
]

For (d_h\ge2), this is

[
\sqrt{d_h},s_\infty^2(d_h-1).
]

Sharpness:

* Near (r\kappa=0), take one vector arbitrarily small but nonzero and aligned with the other in the max-scale coordinate; the error approaches (\sqrt{d_h}s_\infty^2).
* At (r\kappa=d_h), choose all Q columns identical to the hidden vector and all K columns identical to the hidden vector, with (s_m=s_\infty). Then (q=k=(1,\dots,1)), so (r=\kappa=\sqrt{d_h}), and the error is (\sqrt{d_h}s_\infty^2(d_h-1)).

This is not a tight small approximation in the worst case. It is the exact reason Q/K normalization remains part of enGPT.

---

# 14. Final implementation contract

An implementation is enGPT if and only if it satisfies all of the following.

1. It stores hidden state as ((Y,\rho)), with (H=Y\oslash\rho) equal to nGPT’s hidden state.

2. It never feeds raw (Y) into a non-scale-invariant operation without consuming (\rho). In particular:
   [
   V=(YW_V)\oslash\rho,
   ]
   [
   u=(YW_u)\oslash\rho,
   ]
   [
   \nu=(YW_\nu)\oslash\rho.
   ]

3. It may use raw (YW_Q,YW_K) before Q/K normalization only because the per-token positive radial scale cancels exactly in Q/K normalization.

4. It computes exact Q/K norm denominators per token/head.

5. It computes attention scores with (\sqrt{d_h}) scaling and the same learned (s_{qk}) factors as nGPT.

6. It computes attention values, MLP SwiGLU, output projection, down projection, logits, and sampling from exactly the same represented hidden vector (H=Y\oslash\rho) that nGPT would use.

7. It replaces each explicit pair
   [
   N(b),
   \qquad
   N((1-\alpha)\odot h+\alpha\odot N(b))
   ]
   by the carried residual primitive
   [
   u=(1-\alpha)c_b,y+\alpha\rho,b,
   ]
   [
   \rho^+=\max(\lVert u\rVert_2,\varepsilon_r\rho c_b),
   ]
   with the same guard convention as the reference.

8. It normalizes/retracts all nGPT parameter vectors on the correct axes after every update.

9. It uses no weight decay on normalized vectors.

10. Its output head computes
    [
    z_v=s_{z,v}\langle E_{{\rm out},v:},Y/\rho\rangle
    ]
    exactly, with no final RMSNorm/LayerNorm.

Under that contract, enGPT is not a new model. It is nGPT with its normalization algebra moved into GPU-friendly carried scalars and fused residual reductions.

[1]: https://arxiv.org/html/2410.01131v2 "nGPT: Normalized Transformer with Representation Learning on the Hypersphere"
[2]: https://raw.githubusercontent.com/NVIDIA/ngpt/main/train.py "raw.githubusercontent.com"
