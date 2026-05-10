from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import EfficientNGPT, ModelConfig, build_ngpt_adamw


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_batch(step: int, cfg: ModelConfig, batch_size: int, device: torch.device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1000 + step)
    x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), generator=generator)
    x = x.to(device)
    y = (x + 1) % cfg.vocab_size
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(7)
    cfg = ModelConfig(
        vocab_size=128,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=64,
        mlp_ratio=2.0,
        alpha_init=0.2,
    )
    model = EfficientNGPT(cfg).to(device)
    opt = build_ngpt_adamw(model, lr=2e-3)

    losses: list[float] = []
    for step in range(args.steps):
        x, y = make_batch(step, cfg, batch_size=4, device=device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

    print(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "losses": losses,
                "parameter_norms": model.parameter_norm_report(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
