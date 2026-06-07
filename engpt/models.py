from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from .config import ModelConfig
from .kernels import (
    apply_rope,
    carried_up_gate_swiglu,
    carried_residual_gauge,
    causal_sdpa,
    gpt_attention_scale,
    ngpt_attention_scale,
    normalize_columns_,
    normalize_last_dim,
    normalize_rows_,
    qkv_postprocess_from_carried,
    reference_residual,
    rotary_frequencies,
    scaled_logits_from_carried,
)


def _ngpt_base_scale(cfg: ModelConfig) -> float:
    return 1.0 / math.sqrt(float(cfg.n_embd))


class GPTBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.up_gate = nn.Linear(cfg.n_embd, 2 * cfg.mlp_width, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.mlp_width, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        residual = x
        x_norm = self.ln_1(x)
        qkv = self.qkv(x_norm).view(
            bsz, seq_len, 3, self.cfg.n_head, self.cfg.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = causal_sdpa(
            q,
            k,
            v,
            scale=gpt_attention_scale(self.cfg.head_dim),
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        x = residual + self.dropout(self.out_proj(attn.reshape(bsz, seq_len, -1)))

        residual = x
        u, gate = self.up_gate(self.ln_2(x)).chunk(2, dim=-1)
        x = residual + self.dropout(self.down_proj(u * F.silu(gate)))
        return x


class GPTBaseline(nn.Module):
    """Vanilla pre-LN causal GPT trained with AdamW."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.size(1) > self.cfg.block_size:
            raise ValueError("sequence length exceeds block_size")
        cos, sin = rotary_frequencies(
            idx.size(1),
            self.cfg.head_dim,
            device=idx.device,
            dtype=self.token_emb.weight.dtype,
            theta=self.cfg.rope_theta,
        )
        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class EfficientNGPTBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.up_gate = nn.Linear(cfg.n_embd, 2 * cfg.mlp_width, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_width, cfg.n_embd, bias=False)
        base_scale = _ngpt_base_scale(cfg)
        self.alpha_init_value = cfg.alpha_init
        self.scalar_init_scaling = base_scale
        self.alpha_attn_raw = nn.Parameter(torch.full((cfg.n_embd,), base_scale))
        self.alpha_mlp_raw = nn.Parameter(torch.full((cfg.n_embd,), base_scale))
        self.s_qk = nn.Parameter(torch.full((cfg.n_head, cfg.head_dim), base_scale))
        self.s_u = nn.Parameter(torch.ones(cfg.mlp_width))
        self.s_gate = nn.Parameter(torch.ones(cfg.mlp_width))

    @property
    def alpha_attn(self) -> Tensor:
        return self.alpha_attn_raw.abs() * (self.alpha_init_value / self.scalar_init_scaling)

    @property
    def alpha_mlp(self) -> Tensor:
        return self.alpha_mlp_raw.abs() * (self.alpha_init_value / self.scalar_init_scaling)

    @property
    def qk_scale(self) -> Tensor:
        return self.s_qk * (1.0 / self.scalar_init_scaling)

    def forward(self, y: Tensor, rho: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, seq_len, _ = y.shape
        qkv = self.qkv(y).view(bsz, seq_len, 3, self.cfg.n_head, self.cfg.head_dim)
        q, k, v = qkv_postprocess_from_carried(qkv, rho, cos, sin, self.qk_scale, self.cfg.norm_eps)

        attn = causal_sdpa(
            q,
            k,
            v,
            scale=ngpt_attention_scale(self.cfg.head_dim),
            dropout_p=0.0,
        )
        branch = self.out_proj(attn.reshape(bsz, seq_len, -1))
        y, rho = carried_residual_gauge(
            y,
            rho,
            branch,
            self.alpha_attn,
            self.cfg.norm_eps,
            self.cfg.norm_eps,
            self.cfg.carried_gauge_max,
        )

        hidden = carried_up_gate_swiglu(
            y,
            rho,
            self.up_gate.weight,
            self.s_u,
            self.s_gate,
            sqrt_d=math.sqrt(float(self.cfg.n_embd)),
            scale_u_by_sqrt_d=self.cfg.scale_mlp_u_by_sqrt_d,
        )
        branch = self.down_proj(hidden)
        y, rho = carried_residual_gauge(
            y,
            rho,
            branch,
            self.alpha_mlp,
            self.cfg.norm_eps,
            self.cfg.norm_eps,
            self.cfg.carried_gauge_max,
        )
        return y, rho

    def forward_reference(self, h: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = h.shape
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.cfg.n_head, self.cfg.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = normalize_last_dim(apply_rope(q, cos, sin), self.cfg.norm_eps)
        k = normalize_last_dim(apply_rope(k, cos, sin), self.cfg.norm_eps)
        scale = self.qk_scale.view(1, 1, self.cfg.n_head, self.cfg.head_dim)
        q = q * scale
        k = k * scale
        attn = causal_sdpa(
            q,
            k,
            v,
            scale=ngpt_attention_scale(self.cfg.head_dim),
            dropout_p=0.0,
        )
        branch = self.out_proj(attn.reshape(bsz, seq_len, -1))
        h = reference_residual(
            h,
            branch,
            self.alpha_attn,
            self.cfg.norm_eps,
            self.cfg.norm_eps,
        )

        u, gate = self.up_gate(h).chunk(2, dim=-1)
        u = u * self.s_u.view(1, 1, -1)
        if self.cfg.scale_mlp_u_by_sqrt_d:
            u = u * math.sqrt(float(self.cfg.n_embd))
        gate = gate * self.s_gate.view(1, 1, -1) * math.sqrt(float(self.cfg.n_embd))
        branch = self.down_proj(u * F.silu(gate))
        return reference_residual(
            h,
            branch,
            self.alpha_mlp,
            self.cfg.norm_eps,
            self.cfg.norm_eps,
        )

    @torch.no_grad()
    def project_parameters_(self, eps: float = 1e-12) -> None:
        qkv = self.qkv.weight
        normalize_rows_(qkv, eps)
        normalize_columns_(self.out_proj.weight, eps)

        normalize_rows_(self.up_gate.weight, eps)
        normalize_columns_(self.down_proj.weight, eps)


class EfficientNGPT(nn.Module):
    """Exact carried-radius nGPT implementation from `efficient-ngpt.md`."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([EfficientNGPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.output_emb = nn.Parameter(torch.empty(cfg.vocab_size, cfg.n_embd))
        self.scalar_init_scaling = _ngpt_base_scale(cfg)
        self.logit_scale_init_value = cfg.logit_scale_init
        self.s_z = nn.Parameter(torch.full((cfg.vocab_size,), self.scalar_init_scaling))
        self.apply(self._init_weights)
        nn.init.normal_(self.output_emb, mean=0.0, std=0.02)
        self.project_parameters_()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        return rotary_frequencies(
            seq_len,
            self.cfg.head_dim,
            device=device,
            dtype=dtype,
            theta=self.cfg.rope_theta,
        )

    def forward(
        self, idx: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y = self.token_emb(idx)
        rho = torch.ones(idx.shape, device=idx.device, dtype=y.dtype)
        return self.forward_carried(y, rho, targets)

    def forward_carried(
        self,
        y: Tensor,
        rho: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if y.size(1) > self.cfg.block_size:
            raise ValueError("sequence length exceeds block_size")
        if rho.shape != y.shape[:2]:
            raise ValueError("rho must have shape [batch, sequence]")
        cos, sin = self._rope(y.size(1), y.device, self.token_emb.weight.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                y, rho = checkpoint(
                    lambda yy, rr, bb=block: bb(yy, rr, cos, sin),
                    y,
                    rho,
                    use_reentrant=True,
                )
            else:
                y, rho = block(y, rho, cos, sin)
        logits = scaled_logits_from_carried(y, rho, self.output_emb, self.logit_scale)
        return logits, self.loss_from_logits(logits, targets)

    def carried_embedding_bags(self, idx: Tensor, bag_size: int, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
        if bag_size < 1:
            raise ValueError("bag_size must be >= 1")
        if idx.size(1) % bag_size != 0:
            raise ValueError("sequence length must be divisible by bag_size")
        emb = self.token_emb(idx)
        bsz, raw_len, dim = emb.shape
        y = emb.view(bsz, raw_len // bag_size, bag_size, dim).sum(dim=2)
        rho = torch.linalg.vector_norm(y, ord=2, dim=-1).clamp_min(eps)
        return y, rho

    def forward_tst_superposition(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        *,
        bag_size: int,
        eps: float = 1e-12,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        y, rho = self.carried_embedding_bags(idx, bag_size=bag_size, eps=eps)
        bag_targets = None
        if targets is not None:
            if targets.ndim == 2:
                if targets.shape[0] != idx.shape[0] or targets.shape[1] != y.shape[1] * bag_size:
                    raise ValueError("2D TST targets must have shape [batch, latent_seq * bag_size]")
                bag_targets = targets.view(targets.shape[0], y.shape[1], bag_size)
            elif targets.ndim == 3:
                if targets.shape[:2] != y.shape[:2] or targets.shape[2] != bag_size:
                    raise ValueError("3D TST targets must have shape [batch, latent_seq, bag_size]")
                bag_targets = targets
            else:
                raise ValueError("TST targets must be 2D or 3D")
        return self.forward_carried(y, rho, bag_targets)

    @staticmethod
    def loss_from_logits(logits: Tensor, targets: Optional[Tensor]) -> Optional[Tensor]:
        loss = None
        if targets is not None:
            if targets.ndim == logits.ndim - 1:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            elif targets.ndim == logits.ndim:
                log_probs = F.log_softmax(logits.float(), dim=-1)
                loss = -log_probs.gather(-1, targets.long()).mean()
            else:
                raise ValueError("targets must be [batch, sequence] or [batch, sequence, bag_size]")
        return loss

    def forward_reference(
        self, idx: Tensor, targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if idx.size(1) > self.cfg.block_size:
            raise ValueError("sequence length exceeds block_size")
        cos, sin = self._rope(idx.size(1), idx.device, self.token_emb.weight.dtype)
        h = normalize_last_dim(self.token_emb(idx), self.cfg.norm_eps)
        for block in self.blocks:
            h = block.forward_reference(h, cos, sin)
        logits = F.linear(h, self.output_emb * self.logit_scale.unsqueeze(-1))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def project_parameters_(self, eps: float = 1e-12) -> None:
        normalize_rows_(self.token_emb.weight, eps)
        normalize_rows_(self.output_emb, eps)
        for block in self.blocks:
            block.project_parameters_(eps)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    @property
    def logit_scale(self) -> Tensor:
        return self.s_z * (self.logit_scale_init_value / self.scalar_init_scaling)

    def parameter_norm_report(self) -> Dict[str, float]:
        with torch.no_grad():
            max_err = 0.0
            for name, norms in self._normalized_parameter_norms():
                err = (norms - 1.0).abs().max().item()
                max_err = max(max_err, err)
            return {"max_unit_norm_error": max_err}

    def _normalized_parameter_norms(self):
        yield "token_emb", torch.linalg.vector_norm(self.token_emb.weight, dim=-1)
        yield "output_emb", torch.linalg.vector_norm(self.output_emb, dim=-1)
        for i, block in enumerate(self.blocks):
            qkv = block.qkv.weight
            d = self.cfg.n_embd
            yield f"blocks.{i}.q", torch.linalg.vector_norm(qkv[:d], dim=-1)
            yield f"blocks.{i}.k", torch.linalg.vector_norm(qkv[d : 2 * d], dim=-1)
            yield f"blocks.{i}.v", torch.linalg.vector_norm(qkv[2 * d :], dim=-1)
            yield f"blocks.{i}.out", torch.linalg.vector_norm(block.out_proj.weight, dim=0)
            ff = self.cfg.mlp_width
            yield f"blocks.{i}.u", torch.linalg.vector_norm(block.up_gate.weight[:ff], dim=-1)
            yield f"blocks.{i}.gate", torch.linalg.vector_norm(block.up_gate.weight[ff:], dim=-1)
            yield f"blocks.{i}.down", torch.linalg.vector_norm(block.down_proj.weight, dim=0)
