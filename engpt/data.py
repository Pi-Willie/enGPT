from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenDataConfig:
    path: Path
    block_size: int
    batch_size: int
    device: torch.device
    split: str = "train"
    train_fraction: float = 0.98
    seed: int = 1234


class MemmapTokenBatches:
    """Random next-token batches from a token-id memmap."""

    def __init__(self, cfg: TokenDataConfig) -> None:
        self.cfg = cfg
        self.tokens = np.memmap(cfg.path, dtype=np.uint16, mode="r")
        if len(self.tokens) < cfg.block_size + 2:
            raise ValueError("token file is too small for the requested block size")
        split_at = int(len(self.tokens) * cfg.train_fraction)
        if cfg.split == "train":
            self.start = 0
            self.end = split_at
        elif cfg.split == "val":
            self.start = max(0, split_at - cfg.block_size - 1)
            self.end = len(self.tokens)
        else:
            raise ValueError("split must be 'train' or 'val'")
        self.rng = np.random.default_rng(cfg.seed + (0 if cfg.split == "train" else 1))

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def next_batch(self) -> Tuple[Tensor, Tensor]:
        high = self.end - self.cfg.block_size - 1
        if high <= self.start:
            raise ValueError("split is too small for a batch")
        offsets = self.rng.integers(
            self.start,
            high,
            size=self.cfg.batch_size,
            endpoint=False,
            dtype=np.int64,
        )
        x = np.stack([self.tokens[i : i + self.cfg.block_size] for i in offsets])
        y = np.stack([self.tokens[i + 1 : i + self.cfg.block_size + 1] for i in offsets])
        x_t = torch.from_numpy(x.astype(np.int64, copy=False)).to(self.cfg.device, non_blocking=True)
        y_t = torch.from_numpy(y.astype(np.int64, copy=False)).to(self.cfg.device, non_blocking=True)
        return x_t, y_t

    def infinite(self) -> Iterator[Tuple[Tensor, Tensor]]:
        while True:
            yield self.next_batch()
