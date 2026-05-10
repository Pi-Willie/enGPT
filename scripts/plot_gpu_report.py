from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(values, window: int):
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="runs/gpu_report.json")
    parser.add_argument("--out", default="runs/gpu_report_loss.png")
    parser.add_argument("--smooth", type=int, default=50)
    args = parser.parse_args()

    report_path = pathlib.Path(args.report)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = json.load(report_path.open())
    gpt = report["gpt"]["train"]["losses"]
    engpt = report["engpt"]["train"]["losses"]
    steps = list(range(1, len(gpt) + 1))
    plt.figure(figsize=(10, 5.8), dpi=160)
    plt.plot(steps, gpt, color="#64748b", alpha=0.20, linewidth=0.8, label="GPT raw")
    plt.plot(steps, engpt, color="#14b8a6", alpha=0.20, linewidth=0.8, label="enGPT raw")
    plt.plot(steps, moving_average(gpt, args.smooth), color="#334155", linewidth=2.0, label=f"GPT {args.smooth}-step MA")
    plt.plot(steps, moving_average(engpt, args.smooth), color="#0f766e", linewidth=2.0, label=f"enGPT {args.smooth}-step MA")
    plt.title("FineWeb GPT-2 Tokenized 10M-Token Training Loss")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True, color="#d4d4d8", alpha=0.55, linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
