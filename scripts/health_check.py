from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, Tuple

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import (
    EfficientNGPT,
    GPTBaseline,
    ModelConfig,
    build_gpt_adamw,
    build_ngpt_adamw,
    project_ngpt_parameters_,
)


def make_batch(
    step: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(1234 + step)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, device=device)
    y = (x + 1) % vocab_size
    return x, y


def train_steps(model, opt, cfg, *, steps: int, batch_size: int, seq_len: int, device, ngpt: bool):
    losses = []
    model.train()
    for step in range(steps):
        x, y = make_batch(step, batch_size, seq_len, cfg.vocab_size, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ngpt:
            project_ngpt_parameters_(model)
        losses.append(float(loss.detach().cpu()))
    return losses


@torch.no_grad()
def eval_loss(model, cfg, *, batches: int, batch_size: int, seq_len: int, device) -> float:
    model.eval()
    total = 0.0
    for i in range(batches):
        x, y = make_batch(10_000 + i, batch_size, seq_len, cfg.vocab_size, device)
        _, loss = model(x, y)
        total += float(loss.detach().cpu())
    return total / batches


@torch.no_grad()
def benchmark(model, cfg, *, batch_size: int, seq_len: int, iters: int, device) -> float:
    model.eval()
    x, _ = make_batch(999, batch_size, seq_len, cfg.vocab_size, device)
    for _ in range(3):
        model(x)
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    elapsed = time.perf_counter() - start
    return batch_size * seq_len * iters / elapsed


def equivalence_check(model: EfficientNGPT, cfg: ModelConfig, device) -> Dict[str, float]:
    model.eval()
    x, y = make_batch(42, 2, min(16, cfg.block_size), cfg.vocab_size, device)
    logits, loss = model(x, y)
    ref_logits, ref_loss = model.forward_reference(x, y)
    return {
        "logit_max_abs_diff": float((logits - ref_logits).abs().max().detach().cpu()),
        "loss_abs_diff": float((loss - ref_loss).abs().detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--bench-iters", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(7)
    cfg = ModelConfig(
        vocab_size=64,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        mlp_ratio=2.0,
        dropout=0.0,
        alpha_init=0.2,
        logit_scale_init=8.0,
    )

    gpt = GPTBaseline(cfg).to(device)
    engpt = EfficientNGPT(cfg).to(device)
    gpt_opt = build_gpt_adamw(gpt, lr=2e-3, weight_decay=0.1)
    engpt_opt = build_ngpt_adamw(engpt, lr=2e-3)

    eq = equivalence_check(engpt, cfg, device)
    gpt_initial = eval_loss(gpt, cfg, batches=2, batch_size=4, seq_len=cfg.block_size, device=device)
    engpt_initial = eval_loss(engpt, cfg, batches=2, batch_size=4, seq_len=cfg.block_size, device=device)
    gpt_losses = train_steps(
        gpt,
        gpt_opt,
        cfg,
        steps=args.steps,
        batch_size=4,
        seq_len=cfg.block_size,
        device=device,
        ngpt=False,
    )
    engpt_losses = train_steps(
        engpt,
        engpt_opt,
        cfg,
        steps=args.steps,
        batch_size=4,
        seq_len=cfg.block_size,
        device=device,
        ngpt=True,
    )
    gpt_eval = eval_loss(gpt, cfg, batches=3, batch_size=4, seq_len=cfg.block_size, device=device)
    engpt_eval = eval_loss(engpt, cfg, batches=3, batch_size=4, seq_len=cfg.block_size, device=device)
    gpt_tps = benchmark(gpt, cfg, batch_size=4, seq_len=cfg.block_size, iters=args.bench_iters, device=device)
    engpt_tps = benchmark(
        engpt, cfg, batch_size=4, seq_len=cfg.block_size, iters=args.bench_iters, device=device
    )
    ratio = engpt_tps / gpt_tps
    report = {
        "config": asdict(cfg),
        "equivalence": eq,
        "initial_loss": {"gpt": gpt_initial, "engpt": engpt_initial},
        "train_loss_last": {"gpt": gpt_losses[-1], "engpt": engpt_losses[-1]},
        "eval_loss": {"gpt": gpt_eval, "engpt": engpt_eval},
        "tok_per_sec": {"gpt": gpt_tps, "engpt": engpt_tps, "ratio": ratio},
        "parameter_norms": engpt.parameter_norm_report(),
        "pass": {
            "equivalence": eq["logit_max_abs_diff"] < 2e-4,
            "engpt_beats_gpt_loss": engpt_eval < gpt_eval,
            "throughput_ratio_ge_0_60": ratio >= 0.60,
            "unit_norms": engpt.parameter_norm_report()["max_unit_norm_error"] < 2e-5,
        },
    }
    report["pass"]["all"] = all(report["pass"].values())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    raise SystemExit(0 if report["pass"]["all"] else 1)


if __name__ == "__main__":
    main()
