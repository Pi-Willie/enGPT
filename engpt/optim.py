from __future__ import annotations

from typing import List

import torch
from torch import nn

from .kernels import project_column_grad_, project_row_grad_
from .models import EfficientNGPT, GPTBaseline


def build_gpt_adamw(
    model: GPTBaseline,
    *,
    lr: float = 3e-4,
    betas=(0.9, 0.95),
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "token_emb" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


class NGPTAdamW(torch.optim.AdamW):
    """AdamW with nGPT tangent-gradient projection and exact retraction.

    This is the conservative default optimizer for the PyTorch implementation:
    it uses AdamW moments, no weight decay, projects gradients for normalized
    vectors onto their product-of-spheres tangent spaces, and retracts all nGPT
    normalized vectors onto the correct axes after every step.
    """

    def __init__(
        self,
        model: EfficientNGPT,
        *,
        lr: float = 3e-4,
        betas=(0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        super().__init__(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=0.0)

    def step(self, closure=None):  # type: ignore[override]
        project_ngpt_gradients_(self.model)
        loss = super().step(closure=closure)
        project_ngpt_parameters_(self.model)
        return loss


def build_ngpt_adamw(
    model: EfficientNGPT,
    *,
    lr: float = 3e-4,
    betas=(0.9, 0.95),
) -> NGPTAdamW:
    return NGPTAdamW(model, lr=lr, betas=betas)


@torch.no_grad()
def _project_row_grad_(weight: torch.Tensor, grad: torch.Tensor) -> None:
    project_row_grad_(weight, grad)


@torch.no_grad()
def _project_col_grad_(weight: torch.Tensor, grad: torch.Tensor) -> None:
    project_column_grad_(weight, grad)


@torch.no_grad()
def project_ngpt_gradients_(model: EfficientNGPT) -> None:
    if model.token_emb.weight.grad is not None:
        _project_row_grad_(model.token_emb.weight, model.token_emb.weight.grad)
    if model.output_emb.grad is not None:
        _project_row_grad_(model.output_emb, model.output_emb.grad)

    for block in model.blocks:
        qkv = block.qkv.weight
        if qkv.grad is not None:
            _project_row_grad_(qkv, qkv.grad)

        if block.out_proj.weight.grad is not None:
            _project_col_grad_(block.out_proj.weight, block.out_proj.weight.grad)

        up_gate = block.up_gate.weight
        if up_gate.grad is not None:
            _project_row_grad_(up_gate, up_gate.grad)

        if block.down_proj.weight.grad is not None:
            _project_col_grad_(block.down_proj.weight, block.down_proj.weight.grad)


@torch.no_grad()
def project_ngpt_parameters_(model: EfficientNGPT, eps: float = 1e-12) -> None:
    model.project_parameters_(eps=eps)
