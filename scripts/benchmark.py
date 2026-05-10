from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import EfficientNGPT, GPTBaseline, ModelConfig


@torch.no_grad()
def bench(model, idx, iters: int) -> float:
    model.eval()
    for _ in range(5):
        model(idx)
    if idx.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(idx)
    if idx.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return idx.numel() * iters / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.dim,
        mlp_ratio=4.0,
        dropout=0.0,
    )
    idx = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
    gpt = GPTBaseline(cfg).to(device)
    engpt = EfficientNGPT(cfg).to(device)
    if args.compile:
        gpt = torch.compile(gpt)
        engpt = torch.compile(engpt)
    gpt_tps = bench(gpt, idx, args.iters)
    engpt_tps = bench(engpt, idx, args.iters)
    print(
        json.dumps(
            {
                "device": str(device),
                "gpt_tok_per_sec": gpt_tps,
                "engpt_tok_per_sec": engpt_tps,
                "ratio": engpt_tps / gpt_tps,
                "meets_60_percent": engpt_tps / gpt_tps >= 0.60,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
