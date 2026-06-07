from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import EfficientNGPT, ModelConfig, build_ngpt_adamw
from engpt.data import MemmapTokenBatches, TokenDataConfig


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


class TSTBagBatches:
    def __init__(
        self,
        path: pathlib.Path,
        *,
        latent_seq_len: int,
        bag_size: int,
        batch_size: int,
        device: torch.device,
        split: str,
        train_fraction: float = 0.98,
        seed: int = 1234,
    ) -> None:
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.latent_seq_len = latent_seq_len
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.device = device
        self.raw_input_len = latent_seq_len * bag_size
        self.raw_span_len = (latent_seq_len + 1) * bag_size
        split_at = int(len(self.tokens) * train_fraction)
        if split == "train":
            self.start = 0
            self.end = split_at
        elif split == "val":
            self.start = max(0, split_at - self.raw_span_len)
            self.end = len(self.tokens)
        else:
            raise ValueError("split must be train or val")
        if self.end - self.start <= self.raw_span_len:
            raise ValueError("token file split is too small for TST batches")
        self.rng = np.random.default_rng(seed + (0 if split == "train" else 1))

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        high = self.end - self.raw_span_len
        offsets = self.rng.integers(self.start, high, size=self.batch_size, endpoint=False, dtype=np.int64)
        x = np.stack([self.tokens[i : i + self.raw_input_len] for i in offsets])
        y = np.stack([self.tokens[i + self.bag_size : i + self.bag_size + self.raw_input_len] for i in offsets])
        x_t = torch.from_numpy(x.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
        y_t = torch.from_numpy(y.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
        return x_t, y_t


def make_next_token_batches(
    token_path: pathlib.Path,
    cfg: ModelConfig,
    *,
    batch_size: int,
    device: torch.device,
    split: str,
    seed: int,
) -> MemmapTokenBatches:
    return MemmapTokenBatches(
        TokenDataConfig(
            path=token_path,
            block_size=cfg.block_size,
            batch_size=batch_size,
            device=device,
            split=split,
            seed=seed,
        )
    )


@torch.no_grad()
def eval_next_token(
    model: EfficientNGPT,
    batches: MemmapTokenBatches,
    *,
    iters: int,
    dtype: torch.dtype,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = batches.next_batch()
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def train_standard(
    model: EfficientNGPT,
    token_path: pathlib.Path,
    cfg: ModelConfig,
    *,
    seconds: float,
    batch_size: int,
    lr: float,
    dtype: torch.dtype,
    device: torch.device,
    eval_interval_sec: float,
    eval_iters: int,
) -> Dict[str, object]:
    opt = build_ngpt_adamw(model, lr=lr)
    train_batches = make_next_token_batches(token_path, cfg, batch_size=batch_size, device=device, split="train", seed=777)
    eval_batches = make_next_token_batches(token_path, cfg, batch_size=batch_size, device=device, split="val", seed=888)
    records: List[Dict[str, float | int | str]] = []
    eval_records: List[Dict[str, float | int | str]] = []
    start = time.perf_counter()
    next_eval = 0.0
    step = 0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds and step > 0:
            break
        x, y = train_batches.next_batch()
        opt.zero_grad(set_to_none=True)
        synchronize(device)
        step_start = time.perf_counter()
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        records.append(
            {
                "step": step,
                "clock_seconds": elapsed,
                "phase": "standard",
                "loss": float(loss.detach().cpu()),
                "step_seconds": time.perf_counter() - step_start,
                "predicted_tokens": batch_size * cfg.block_size,
            }
        )
        if elapsed >= next_eval:
            eval_records.append(
                {
                    "step": step,
                    "clock_seconds": elapsed,
                    "phase": "standard",
                    "next_token_eval_loss": eval_next_token(
                        model, eval_batches, iters=eval_iters, dtype=dtype, device=device
                    ),
                }
            )
            next_eval += eval_interval_sec
        step += 1
    return {"train": records, "eval": eval_records}


def train_tst(
    model: EfficientNGPT,
    token_path: pathlib.Path,
    cfg: ModelConfig,
    *,
    seconds: float,
    superposition_fraction: float,
    bag_size: int,
    batch_size: int,
    lr: float,
    dtype: torch.dtype,
    device: torch.device,
    eval_interval_sec: float,
    eval_iters: int,
) -> Dict[str, object]:
    opt = build_ngpt_adamw(model, lr=lr)
    tst_batches = TSTBagBatches(
        token_path,
        latent_seq_len=cfg.block_size,
        bag_size=bag_size,
        batch_size=batch_size,
        device=device,
        split="train",
        seed=777,
    )
    recovery_batches = make_next_token_batches(
        token_path, cfg, batch_size=batch_size, device=device, split="train", seed=777
    )
    eval_batches = make_next_token_batches(token_path, cfg, batch_size=batch_size, device=device, split="val", seed=888)
    records: List[Dict[str, float | int | str]] = []
    eval_records: List[Dict[str, float | int | str]] = []
    start = time.perf_counter()
    transition_sec = seconds * superposition_fraction
    next_eval = 0.0
    step = 0
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= seconds and step > 0:
            break
        phase = "superposition" if elapsed < transition_sec else "recovery"
        if phase == "superposition":
            x, y = tst_batches.next_batch()
        else:
            x, y = recovery_batches.next_batch()
        opt.zero_grad(set_to_none=True)
        synchronize(device)
        step_start = time.perf_counter()
        with autocast_context(device, dtype):
            if phase == "superposition":
                _, loss = model.forward_tst_superposition(x, y, bag_size=bag_size)
                predicted_tokens = batch_size * cfg.block_size * bag_size
            else:
                _, loss = model(x, y)
                predicted_tokens = batch_size * cfg.block_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        records.append(
            {
                "step": step,
                "clock_seconds": elapsed,
                "phase": phase,
                "loss": float(loss.detach().cpu()),
                "step_seconds": time.perf_counter() - step_start,
                "predicted_tokens": predicted_tokens,
            }
        )
        if elapsed >= next_eval:
            eval_records.append(
                {
                    "step": step,
                    "clock_seconds": elapsed,
                    "phase": phase,
                    "next_token_eval_loss": eval_next_token(
                        model, eval_batches, iters=eval_iters, dtype=dtype, device=device
                    ),
                }
            )
            next_eval += eval_interval_sec
        step += 1
    return {"train": records, "eval": eval_records, "transition_sec": transition_sec}


def plot_report(report: Dict[str, object], path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), dpi=145)
    for name, color in [("standard", "#2563eb"), ("tst", "#dc2626")]:
        rows = report["runs"][name]["train"]
        x = [r["clock_seconds"] for r in rows]
        y = [r["loss"] for r in rows]
        axes[0].plot(x, y, label=name, color=color, alpha=0.88)
    transition = report["runs"]["tst"].get("transition_sec")
    if transition is not None:
        axes[0].axvline(float(transition), color="#991b1b", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("Training objective")
    axes[0].set_xlabel("seconds")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for name, color in [("standard", "#2563eb"), ("tst", "#dc2626")]:
        rows = report["runs"][name]["eval"]
        x = [r["clock_seconds"] for r in rows]
        y = [r["next_token_eval_loss"] for r in rows]
        axes[1].plot(x, y, marker="o", label=name, color=color)
    if transition is not None:
        axes[1].axvline(float(transition), color="#991b1b", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Next-token eval loss")
    axes[1].set_xlabel("seconds")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def summarize_run(run: Dict[str, object]) -> Dict[str, float | int | str]:
    train = run["train"]
    eval_rows = run["eval"]
    elapsed = float(train[-1]["clock_seconds"]) if train else 0.0
    predicted = sum(int(r["predicted_tokens"]) for r in train)
    return {
        "steps": len(train),
        "elapsed_sec": elapsed,
        "predicted_tokens": predicted,
        "predicted_tok_per_sec": predicted / elapsed if elapsed > 0 else 0.0,
        "first_train_loss": float(train[0]["loss"]),
        "final_train_loss": float(train[-1]["loss"]),
        "first_eval_loss": float(eval_rows[0]["next_token_eval_loss"]),
        "final_eval_loss": float(eval_rows[-1]["next_token_eval_loss"]),
        "eval_loss_delta": float(eval_rows[-1]["next_token_eval_loss"]) - float(eval_rows[0]["next_token_eval_loss"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", default="runs/tst_benchmark/report.json")
    parser.add_argument("--plot", default="runs/tst_benchmark/loss_compare.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seconds-per-run", type=float, default=60.0)
    parser.add_argument("--superposition-fraction", type=float, default=0.4)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--eval-interval-sec", type=float, default=10.0)
    parser.add_argument("--eval-iters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.dim,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
    )
    base = EfficientNGPT(cfg).to(device)
    initial = copy.deepcopy(base.state_dict())
    token_path = pathlib.Path(args.tokens)

    standard = EfficientNGPT(cfg).to(device)
    standard.load_state_dict(initial)
    standard_run = train_standard(
        standard,
        token_path,
        cfg,
        seconds=args.seconds_per_run,
        batch_size=args.batch_size,
        lr=args.lr,
        dtype=dtype,
        device=device,
        eval_interval_sec=args.eval_interval_sec,
        eval_iters=args.eval_iters,
    )
    del standard
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tst = EfficientNGPT(cfg).to(device)
    tst.load_state_dict(initial)
    tst_run = train_tst(
        tst,
        token_path,
        cfg,
        seconds=args.seconds_per_run,
        superposition_fraction=args.superposition_fraction,
        bag_size=args.bag_size,
        batch_size=args.batch_size,
        lr=args.lr,
        dtype=dtype,
        device=device,
        eval_interval_sec=args.eval_interval_sec,
        eval_iters=args.eval_iters,
    )

    report = {
        "config": asdict(cfg),
        "args": vars(args),
        "runs": {"standard": standard_run, "tst": tst_run},
        "summary": {"standard": summarize_run(standard_run), "tst": summarize_run(tst_run)},
    }
    report["summary"]["tst_eval_delta_minus_standard"] = (
        report["summary"]["tst"]["eval_loss_delta"] - report["summary"]["standard"]["eval_loss_delta"]
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    plot_report(report, pathlib.Path(args.plot))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
