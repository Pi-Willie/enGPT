from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import EfficientNGPT, GPTBaseline, ModelConfig, build_gpt_adamw, build_ngpt_adamw


def make_batch(
    step: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic next-token toy task with fixed per-step batches."""

    g = torch.Generator(device="cpu")
    g.manual_seed(20_000 + step)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, device=device)
    y = (x + 1) % vocab_size
    return x, y


def moving_average(values: List[float], window: int) -> List[float]:
    out: List[float] = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def train_pair(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
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
    )
    gpt = GPTBaseline(cfg).to(device)
    engpt = EfficientNGPT(cfg).to(device)
    if args.compile:
        gpt = torch.compile(gpt)
        engpt = torch.compile(engpt)

    gpt_opt = build_gpt_adamw(gpt, lr=args.lr, weight_decay=args.gpt_weight_decay)
    engpt_opt = build_ngpt_adamw(engpt, lr=args.lr)

    gpt_losses: List[float] = []
    engpt_losses: List[float] = []
    start = time.perf_counter()
    for step in range(args.steps):
        x, y = make_batch(step, args.batch_size, args.seq_len, cfg.vocab_size, device)

        gpt.train()
        gpt_opt.zero_grad(set_to_none=True)
        _, gpt_loss = gpt(x, y)
        gpt_loss.backward()
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), args.grad_clip)
        gpt_opt.step()

        engpt.train()
        engpt_opt.zero_grad(set_to_none=True)
        _, engpt_loss = engpt(x, y)
        engpt_loss.backward()
        torch.nn.utils.clip_grad_norm_(engpt.parameters(), args.grad_clip)
        engpt_opt.step()

        gpt_losses.append(float(gpt_loss.detach().cpu()))
        engpt_losses.append(float(engpt_loss.detach().cpu()))
        if args.log_every and (step + 1) % args.log_every == 0:
            print(
                f"step {step + 1:4d}/{args.steps}: "
                f"gpt={gpt_losses[-1]:.4f} engpt={engpt_losses[-1]:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    return {
        "config": asdict(cfg),
        "train": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "elapsed_sec": elapsed,
            "gpt_losses": gpt_losses,
            "engpt_losses": engpt_losses,
            "gpt_final": gpt_losses[-1],
            "engpt_final": engpt_losses[-1],
            "gpt_ma_final": moving_average(gpt_losses, args.smooth)[-1],
            "engpt_ma_final": moving_average(engpt_losses, args.smooth)[-1],
        },
    }


def write_outputs(report: Dict[str, object], out_dir: pathlib.Path, smooth: int) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = report["train"]
    gpt_losses = train["gpt_losses"]
    engpt_losses = train["engpt_losses"]
    steps = list(range(1, len(gpt_losses) + 1))
    gpt_ma = moving_average(gpt_losses, smooth)
    engpt_ma = moving_average(engpt_losses, smooth)

    csv_path = out_dir / "loss_history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "gpt_loss", "engpt_loss", f"gpt_ma_{smooth}", f"engpt_ma_{smooth}"])
        writer.writerows(zip(steps, gpt_losses, engpt_losses, gpt_ma, engpt_ma))

    json_path = out_dir / "train_report.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    png_path = out_dir / "loss_graph.png"
    plt.figure(figsize=(10, 5.6), dpi=160)
    plt.plot(steps, gpt_losses, color="#64748b", alpha=0.26, linewidth=1.0, label="GPT raw")
    plt.plot(steps, engpt_losses, color="#14b8a6", alpha=0.26, linewidth=1.0, label="enGPT raw")
    plt.plot(steps, gpt_ma, color="#334155", linewidth=2.0, label=f"GPT {smooth}-step MA")
    plt.plot(steps, engpt_ma, color="#0f766e", linewidth=2.0, label=f"enGPT {smooth}-step MA")
    plt.title("GPT vs enGPT Training Loss")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True, color="#d4d4d8", alpha=0.55, linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "png": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gpt-weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--alpha-init", type=float, default=0.2)
    parser.add_argument("--logit-scale-init", type=float, default=8.0)
    parser.add_argument("--smooth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--out-dir", default="runs/loss_compare")
    args = parser.parse_args()

    report = train_pair(args)
    paths = write_outputs(report, pathlib.Path(args.out_dir), args.smooth)
    train = report["train"]
    print(
        json.dumps(
            {
                "gpt_final": train["gpt_final"],
                "engpt_final": train["engpt_final"],
                "gpt_ma_final": train["gpt_ma_final"],
                "engpt_ma_final": train["engpt_ma_final"],
                "elapsed_sec": train["elapsed_sec"],
                "paths": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
