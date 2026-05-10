"""Efficient carried-radius nGPT and baseline GPT models."""

__version__ = "0.1.0"

from .config import ModelConfig
from .models import EfficientNGPT, GPTBaseline
from .optim import (
    NGPTAdamW,
    build_gpt_adamw,
    build_ngpt_adamw,
    project_ngpt_gradients_,
    project_ngpt_parameters_,
)

__all__ = [
    "EfficientNGPT",
    "GPTBaseline",
    "ModelConfig",
    "NGPTAdamW",
    "__version__",
    "build_gpt_adamw",
    "build_ngpt_adamw",
    "project_ngpt_gradients_",
    "project_ngpt_parameters_",
]
