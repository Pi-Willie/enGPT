from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import EfficientNGPT, GPTBaseline, ModelConfig, build_gpt_adamw, build_ngpt_adamw
from engpt.data import MemmapTokenBatches, TokenDataConfig


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_model(
    label: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    batches: MemmapTokenBatches,
    *,
    steps: int,
    warmup_steps: int,
    lr_schedule_steps: int,
    base_lr: float,
    grad_clip: float,
    dtype: torch.dtype,
    device: torch.device,
    record_step_stats: bool,
) -> Dict[str, object]:
    losses: List[float] = []
    grad_norms: List[float] = []
    step_seconds: List[float] = []
    clock_seconds: List[float] = []
    model.train()
    def set_scheduled_lr(global_step: int) -> None:
        if lr_schedule_steps > 0:
            lr = base_lr * min(1.0, float(global_step + 1) / float(lr_schedule_steps))
            for group in opt.param_groups:
                group["lr"] = lr

    for _ in range(warmup_steps):
        set_scheduled_lr(_)
        x, y = batches.next_batch()
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    synchronize(device)
    reset_peak(device)
    synchronize(device)
    start = time.perf_counter()
    for step in range(steps):
        if record_step_stats:
            synchronize(device)
            step_start = time.perf_counter()
        set_scheduled_lr(warmup_steps + step)
        x, y = batches.next_batch()
        opt.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if record_step_stats:
            synchronize(device)
            step_elapsed = time.perf_counter() - step_start
            step_seconds.append(step_elapsed)
            clock_seconds.append((clock_seconds[-1] if clock_seconds else 0.0) + step_elapsed)
            grad_norms.append(float(grad_norm.detach().cpu()))
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"{label} step {step + 1}/{steps}: loss={losses[-1]:.4f}", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = steps * batches.cfg.batch_size * batches.cfg.block_size
    return {
        "losses": losses,
        "warmup_steps": warmup_steps,
        "lr_schedule_steps": lr_schedule_steps,
        "final_loss": losses[-1],
        "mean_last_10": sum(losses[-min(10, len(losses)) :]) / min(10, len(losses)),
        "elapsed_sec": elapsed,
        "tok_per_sec": tokens / elapsed,
        "peak_memory_gib": peak_gib(device),
        "step_stats_recorded": record_step_stats,
        "grad_norms": grad_norms,
        "step_seconds": step_seconds,
        "clock_seconds": clock_seconds,
    }


@torch.no_grad()
def eval_model(
    model: torch.nn.Module,
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
    return sum(losses) / len(losses)


@torch.no_grad()
def forward_benchmark(
    label: str,
    model: torch.nn.Module,
    batches: MemmapTokenBatches,
    *,
    iters: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    for _ in range(5):
        x, _ = batches.next_batch()
        with autocast_context(device, dtype):
            model(x)
    reset_peak(device)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        x, _ = batches.next_batch()
        with autocast_context(device, dtype):
            model(x)
    synchronize(device)
    elapsed = time.perf_counter() - start
    tokens = iters * batches.cfg.batch_size * batches.cfg.block_size
    out = {"tok_per_sec": tokens / elapsed, "elapsed_sec": elapsed, "peak_memory_gib": peak_gib(device)}
    print(f"{label} forward: {out}", flush=True)
    return out


@torch.no_grad()
def engpt_reference_check(model: EfficientNGPT, batches: MemmapTokenBatches, *, dtype: torch.dtype, device: torch.device):
    model.eval()
    x, y = batches.next_batch()
    with autocast_context(device, dtype):
        logits, loss = model(x, y)
        ref_logits, ref_loss = model.forward_reference(x, y)
    return {
        "logit_max_abs_diff": float((logits - ref_logits).abs().max().detach().float().cpu()),
        "loss_abs_diff": float((loss - ref_loss).abs().detach().float().cpu()),
    }


def parse_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="data/fineweb_gpt2_10m_u16.bin")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--beat-after-step", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gpt-lr", type=float, default=None)
    parser.add_argument("--engpt-lr", type=float, default=8e-3)
    parser.add_argument("--gpt-warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-init", type=float, default=0.05)
    parser.add_argument("--logit-scale-init", type=float, default=8.0)
    parser.add_argument("--no-mlp-u-sqrt-scale", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--engpt-checkpoint", action="store_true")
    parser.add_argument(
        "--record-step-stats",
        action="store_true",
        help="Synchronize measured training steps and record grad norms plus cumulative wall time.",
    )
    parser.add_argument("--out", default="runs/gpu_report.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(123)
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.dim,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        alpha_init=args.alpha_init,
        logit_scale_init=args.logit_scale_init,
        scale_mlp_u_by_sqrt_d=not args.no_mlp_u_sqrt_scale,
    )
    def make_train_batches(seed: int = 777) -> MemmapTokenBatches:
        return MemmapTokenBatches(
            TokenDataConfig(
                path=pathlib.Path(args.tokens),
                block_size=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                split="train",
                seed=seed,
            )
        )

    def make_val_batches(seed: int = 888) -> MemmapTokenBatches:
        return MemmapTokenBatches(
            TokenDataConfig(
                path=pathlib.Path(args.tokens),
                block_size=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                split="val",
                seed=seed,
            )
        )

    # Touch the memmap once before model allocation so file/data errors fail
    # before GPU memory is involved.
    _ = MemmapTokenBatches(
        TokenDataConfig(
            path=pathlib.Path(args.tokens),
            block_size=args.seq_len,
            batch_size=args.batch_size,
            device=device,
            split="train",
            seed=777,
        )
    )

    def make_gpt():
        model = GPTBaseline(cfg).to(device)
        if args.compile:
            kwargs = {} if args.compile_mode == "default" else {"mode": args.compile_mode}
            return model, torch.compile(model, **kwargs)
        return model, model

    def make_engpt():
        model = EfficientNGPT(cfg).to(device)
        model.set_gradient_checkpointing(args.engpt_checkpoint)
        if args.compile:
            kwargs = {} if args.compile_mode == "default" else {"mode": args.compile_mode}
            return model, torch.compile(model, **kwargs)
        return model, model

    clear_device(device)
    gpt_base, gpt = make_gpt()
    gpt_lr = args.gpt_lr or args.lr
    engpt_lr = args.engpt_lr or args.lr
    gpt_opt = build_gpt_adamw(gpt_base, lr=gpt_lr, weight_decay=0.1)
    gpt_train = train_model(
        "gpt",
        gpt,
        gpt_opt,
        make_train_batches(),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr_schedule_steps=args.gpt_warmup_steps,
        base_lr=gpt_lr,
        grad_clip=args.grad_clip,
        dtype=dtype,
        device=device,
        record_step_stats=args.record_step_stats,
    )
    gpt_eval = eval_model(gpt, make_val_batches(), iters=args.eval_iters, dtype=dtype, device=device)
    gpt_forward = forward_benchmark(
        "gpt", gpt, make_val_batches(889), iters=args.bench_iters, dtype=dtype, device=device
    )
    del gpt_opt, gpt, gpt_base
    clear_device(device)

    engpt_base, engpt = make_engpt()
    engpt_opt = build_ngpt_adamw(engpt_base, lr=engpt_lr)
    engpt_train = train_model(
        "engpt",
        engpt,
        engpt_opt,
        make_train_batches(),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        lr_schedule_steps=0,
        base_lr=engpt_lr,
        grad_clip=args.grad_clip,
        dtype=dtype,
        device=device,
        record_step_stats=args.record_step_stats,
    )
    engpt_eval = eval_model(engpt, make_val_batches(), iters=args.eval_iters, dtype=dtype, device=device)
    exactness = engpt_reference_check(engpt_base, make_val_batches(890), dtype=dtype, device=device)
    engpt_forward = forward_benchmark(
        "engpt", engpt, make_val_batches(889), iters=args.bench_iters, dtype=dtype, device=device
    )

    forward_throughput_ratio = engpt_forward["tok_per_sec"] / gpt_forward["tok_per_sec"]
    train_throughput_ratio = engpt_train["tok_per_sec"] / gpt_train["tok_per_sec"]
    memory_ratio = engpt_train["peak_memory_gib"] / max(gpt_train["peak_memory_gib"], 1e-9)
    loss_delta = engpt_eval - gpt_eval
    eval_loss_ratio = engpt_eval / max(gpt_eval, 1e-9)
    beat_start = max(0, min(args.beat_after_step, len(gpt_train["losses"])))
    post_losses = list(zip(gpt_train["losses"][beat_start:], engpt_train["losses"][beat_start:]))
    post_step_win_fraction = (
        sum(1 for g, e in post_losses if e < g) / len(post_losses) if post_losses else 0.0
    )
    post_step_all_wins = bool(post_losses) and all(e < g for g, e in post_losses)
    report = {
        "config": asdict(cfg),
        "device": str(device),
        "dtype": args.dtype,
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "engpt_checkpoint": args.engpt_checkpoint,
        "optimizer": {
            "gpt_lr": gpt_lr,
            "engpt_lr": engpt_lr,
            "gpt_warmup_steps": args.gpt_warmup_steps,
            "engpt_warmup_steps": 0,
        },
        "data": {"tokens": args.tokens, "train_tokens_seen_per_model": args.steps * args.batch_size * args.seq_len},
        "gpt": {"train": gpt_train, "eval_loss": gpt_eval, "forward": gpt_forward},
        "engpt": {
            "train": engpt_train,
            "eval_loss": engpt_eval,
            "forward": engpt_forward,
            "reference_exactness": exactness,
        },
        "comparison": {
            "forward_throughput_ratio": forward_throughput_ratio,
            "train_throughput_ratio": train_throughput_ratio,
            "memory_ratio": memory_ratio,
            "eval_loss_delta_engpt_minus_gpt": loss_delta,
            "eval_loss_ratio_engpt_over_gpt": eval_loss_ratio,
            "beat_after_step": args.beat_after_step,
            "post_step_win_fraction": post_step_win_fraction,
            "post_step_all_wins": post_step_all_wins,
            "forward_throughput_gate_70pct": forward_throughput_ratio >= 0.70,
            "train_throughput_gate_70pct": train_throughput_ratio >= 0.70,
            "memory_gate_within_20pct": memory_ratio <= 1.20,
            "loss_gate_reference_abs_2e_3": exactness["loss_abs_diff"] <= 2e-3,
            "eval_loss_gate_within_5pct_of_gpt": eval_loss_ratio <= 1.05,
            "post_step_all_wins_gate": post_step_all_wins,
        },
    }
    report["comparison"]["all_gates"] = all(
        [
            report["comparison"]["forward_throughput_gate_70pct"],
            report["comparison"]["train_throughput_gate_70pct"],
            report["comparison"]["memory_gate_within_20pct"],
            report["comparison"]["loss_gate_reference_abs_2e_3"],
            report["comparison"]["eval_loss_gate_within_5pct_of_gpt"],
            report["comparison"]["post_step_all_wins_gate"],
        ]
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    print(f"wrote {out}")
    raise SystemExit(0 if report["comparison"]["all_gates"] else 1)


if __name__ == "__main__":
    main()
