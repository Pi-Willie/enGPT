from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("model dimensions must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_width(self) -> int:
        return int(self.mlp_ratio * self.n_embd)
