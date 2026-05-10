from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GPT_COLOR = "#2f3b4f"
NGPT_COLOR = "#087f74"
GRID_COLOR = "#d6dbe3"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_report(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def moving_average(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def step_axis(losses: list[float]) -> list[int]:
    return list(range(1, len(losses) + 1))


def clock_axis(train: dict) -> list[float]:
    losses = train["losses"]
    recorded = train.get("clock_seconds") or []
    if len(recorded) == len(losses):
        return recorded
    if not losses:
        return []
    elapsed = float(train.get("elapsed_sec", 0.0))
    return [elapsed * (i + 1) / len(losses) for i in range(len(losses))]


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.tick_params(axis="both", labelsize=8)


def save_figure(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=360, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_loss_pair(
    reports: list[tuple[str, dict]],
    *,
    x_mode: str,
    out_dir: Path,
    stem: str,
    title: str,
    smooth: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, len(reports), figsize=(7.1, 2.45), sharey=True)
    if len(reports) == 1:
        axes = [axes]
    for ax, (label, report) in zip(axes, reports):
        gpt = report["gpt"]["train"]
        ngpt = report["engpt"]["train"]
        gpt_loss = gpt["losses"]
        ngpt_loss = ngpt["losses"]
        if x_mode == "clock":
            x_gpt = clock_axis(gpt)
            x_ngpt = clock_axis(ngpt)
            xlabel = "wall clock seconds"
        else:
            x_gpt = step_axis(gpt_loss)
            x_ngpt = step_axis(ngpt_loss)
            xlabel = "training step"
        ax.plot(x_gpt, gpt_loss, color=GPT_COLOR, alpha=0.13, linewidth=0.55)
        ax.plot(x_ngpt, ngpt_loss, color=NGPT_COLOR, alpha=0.15, linewidth=0.55)
        ax.plot(x_gpt, moving_average(gpt_loss, smooth), color=GPT_COLOR, linewidth=1.45, label="GPT")
        ax.plot(x_ngpt, moving_average(ngpt_loss, smooth), color=NGPT_COLOR, linewidth=1.45, label="Efficient nGPT")
        ax.set_title(label, pad=4)
        ax.set_xlabel(xlabel)
        style_axes(ax)
        ax.text(
            0.03,
            0.06,
            f"eval GPT {report['gpt']['eval_loss']:.2f}\neval nGPT {report['engpt']['eval_loss']:.2f}",
            transform=ax.transAxes,
            fontsize=6.8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "boxstyle": "round,pad=0.25"},
        )
    axes[0].set_ylabel("cross-entropy loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return save_figure(fig, out_dir, stem)


def plot_depth_eval(reports: list[tuple[str, dict]], out_dir: Path) -> list[Path]:
    labels = [name.split()[0] for name, _ in reports]
    gpt = [r["gpt"]["eval_loss"] for _, r in reports]
    ngpt = [r["engpt"]["eval_loss"] for _, r in reports]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    ax.bar([i - width / 2 for i in x], gpt, width=width, color=GPT_COLOR, label="GPT")
    ax.bar([i + width / 2 for i in x], ngpt, width=width, color=NGPT_COLOR, label="Efficient nGPT")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("eval cross-entropy", fontsize=8)
    ax.set_xlabel("layers", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "depth_eval_loss_bars")


def plot_speed_memory(reports: list[tuple[str, dict]], out_dir: Path) -> list[Path]:
    labels = [name.split()[0] for name, _ in reports]
    train_ratio = [r["comparison"]["train_throughput_ratio"] for _, r in reports]
    memory_ratio = [r["comparison"]["memory_ratio"] for _, r in reports]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.bar([i - width / 2 for i in x], train_ratio, width=width, color=GPT_COLOR, label="train tokens/sec ratio")
    ax.bar([i + width / 2 for i in x], memory_ratio, width=width, color=NGPT_COLOR, label="peak train memory ratio")
    ax.axhline(1.0, color="#8a94a6", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.25, max(memory_ratio) * 1.2))
    ax.set_ylabel("nGPT / GPT", fontsize=8)
    ax.set_xlabel("layers", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "depth_speed_memory_tradeoff")


def plot_trace_loss(report: dict, out_dir: Path, *, x_mode: str, stem: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for key, color, label in [("gpt", GPT_COLOR, "GPT"), ("engpt", NGPT_COLOR, "Efficient nGPT")]:
        train = report[key]["train"]
        losses = train["losses"]
        x = clock_axis(train) if x_mode == "clock" else step_axis(losses)
        ax.plot(x, losses, color=color, alpha=0.16, linewidth=0.55)
        ax.plot(x, moving_average(losses, 20), color=color, linewidth=1.55, label=label)
    ax.set_xlabel("wall clock seconds" if x_mode == "clock" else "training step", fontsize=8)
    ax.set_ylabel("cross-entropy loss", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)


def plot_trace_grad_norm(report: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    plotted = False
    for key, color, label in [("gpt", GPT_COLOR, "GPT"), ("engpt", NGPT_COLOR, "Efficient nGPT")]:
        norms = report[key]["train"].get("grad_norms") or []
        if not norms:
            continue
        ax.plot(step_axis(norms), moving_average(norms, 20), color=color, linewidth=1.55, label=label)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "grad norms not recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("pre-clip grad norm", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "trace_grad_norm")


def plot_step_time(report: dict, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    plotted = False
    for key, color, label in [("gpt", GPT_COLOR, "GPT"), ("engpt", NGPT_COLOR, "Efficient nGPT")]:
        times = report[key]["train"].get("step_seconds") or []
        if not times:
            continue
        ax.plot(step_axis(times), moving_average(times, 20), color=color, linewidth=1.55, label=label)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "step times not recorded", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("seconds per step", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "trace_step_time")


def plot_readme_trace(report: dict, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
    ax = axes[0]
    for key, color, label in [("gpt", GPT_COLOR, "GPT"), ("engpt", NGPT_COLOR, "Efficient nGPT")]:
        train = report[key]["train"]
        losses = train["losses"]
        x = clock_axis(train)
        ax.plot(x, losses, color=color, alpha=0.14, linewidth=0.55)
        ax.plot(x, moving_average(losses, 20), color=color, linewidth=1.55, label=label)
    ax.set_xlabel("wall clock seconds", fontsize=8)
    ax.set_ylabel("cross-entropy loss", fontsize=8)
    ax.set_title("loss against time", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[1]
    for key, color, label in [("gpt", GPT_COLOR, "GPT"), ("engpt", NGPT_COLOR, "Efficient nGPT")]:
        norms = report[key]["train"].get("grad_norms") or []
        if norms:
            ax.plot(step_axis(norms), moving_average(norms, 20), color=color, linewidth=1.55, label=label)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("pre-clip grad norm", fontsize=8)
    ax.set_title("optimization trace", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "readme_trace_dynamics")


def plot_regime_summary(reports: Iterable[dict], out_dir: Path) -> list[Path]:
    rows = []
    for report in reports:
        cfg = report["config"]
        label = f"{cfg['n_layer']}L d{cfg['n_embd']} T{cfg['block_size']}"
        rows.append((label, report["gpt"]["eval_loss"], report["engpt"]["eval_loss"]))
    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(rows)), 2.7))
    x = list(range(len(rows)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], [r[1] for r in rows], width=width, color=GPT_COLOR, label="GPT")
    ax.bar([i + width / 2 for i in x], [r[2] for r in rows], width=width, color=NGPT_COLOR, label="Efficient nGPT")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("eval cross-entropy", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, out_dir, "regime_eval_summary")


def discover_trace_report(reports_dir: Path) -> Optional[Path]:
    candidates = sorted(reports_dir.glob("*trace*.json"))
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="assets/reports")
    parser.add_argument("--out-dir", default="assets/figures")
    parser.add_argument("--trace-report", default=None)
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)
    gallery = out_dir / "gallery"

    depth_inputs = [
        ("64 layers", reports_dir / "gpu_report_depth_64l_d128_s128_gauge_500.json"),
        ("128 layers", reports_dir / "gpu_report_depth_128l_d128_s128_gauge_300.json"),
    ]
    depth_reports = [(label, report) for label, path in depth_inputs if (report := load_report(path)) is not None]
    written: list[Path] = []
    if depth_reports:
        written += plot_loss_pair(
            depth_reports,
            x_mode="step",
            out_dir=out_dir,
            stem="readme_depth_loss",
            title="deep training loss",
            smooth=args.smooth,
        )
        written += plot_loss_pair(
            depth_reports,
            x_mode="clock",
            out_dir=gallery,
            stem="depth_loss_clock",
            title="deep loss against wall clock",
            smooth=args.smooth,
        )
        written += plot_depth_eval(depth_reports, gallery)
        written += plot_speed_memory(depth_reports, gallery)

    trace_path = Path(args.trace_report) if args.trace_report else discover_trace_report(reports_dir)
    trace_report = load_report(trace_path) if trace_path else None
    if trace_report is not None:
        written += plot_readme_trace(trace_report, out_dir)
        written += plot_trace_loss(trace_report, gallery, x_mode="step", stem="trace_loss_step")
        written += plot_trace_loss(trace_report, gallery, x_mode="clock", stem="trace_loss_clock")
        written += plot_trace_grad_norm(trace_report, gallery)
        written += plot_step_time(trace_report, gallery)

    token_eff = load_report(reports_dir / "gpu_report_token_efficiency_lr1e3.json")
    if token_eff is not None:
        written += plot_trace_loss(token_eff, gallery, x_mode="step", stem="token_efficiency_loss_step")

    regime_reports = [
        report
        for path in [
            reports_dir / "gpu_report_regime_2x128_s128_500.json",
            reports_dir / "gpu_report_regime_4x256_s512_500.json",
            reports_dir / "gpu_report_token_efficiency_lr1e3.json",
        ]
        if (report := load_report(path)) is not None
    ]
    if regime_reports:
        written += plot_regime_summary(regime_reports, gallery)

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
