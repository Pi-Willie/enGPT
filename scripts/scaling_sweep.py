from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engpt import EfficientNGPT, ModelConfig, build_ngpt_adamw
from scripts.tst_benchmark import (
    TSTBagBatches,
    autocast_context,
    eval_next_token,
    make_next_token_batches,
    moving_average,
    parse_dtype,
    synchronize,
)


def choose_heads(dim: int, target_head_dim: int = 64) -> int:
    heads = max(1, round(dim / target_head_dim))
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


def estimate_ngpt_params(
    *,
    vocab_size: int,
    layers: int,
    dim: int,
    mlp_ratio: float,
) -> Dict[str, int]:
    mlp_width = int(mlp_ratio * dim)
    embedding = 2 * vocab_size * dim + vocab_size
    per_layer = (
        3 * dim * dim
        + dim * dim
        + 2 * mlp_width * dim
        + mlp_width * dim
        + 3 * dim
        + 2 * mlp_width
    )
    total = embedding + layers * per_layer
    return {
        "embedding_params": int(embedding),
        "nonembed_params": int(layers * per_layer),
        "per_layer_params": int(per_layer),
        "total_params": int(total),
    }


def exact_param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def add_spec(
    specs: Dict[str, Dict[str, object]],
    *,
    family: str,
    layers: int,
    dim: int,
) -> None:
    key = f"L{layers:03d}_D{dim:04d}"
    entry = specs.setdefault(
        key,
        {
            "name": key,
            "layers": layers,
            "dim": dim,
            "families": [],
        },
    )
    families = entry["families"]
    if family not in families:
        families.append(family)


def default_specs() -> List[Dict[str, object]]:
    specs: Dict[str, Dict[str, object]] = {}
    for dim in [128, 192, 256, 384, 512, 640]:
        add_spec(specs, family="width_sweep_L8", layers=8, dim=dim)
    for layers in [2, 4, 8, 16, 32, 48]:
        add_spec(specs, family="depth_sweep_D256", layers=layers, dim=256)
    for layers, dim in [(2, 128), (4, 192), (8, 256), (12, 384), (16, 512), (20, 640)]:
        add_spec(specs, family="diagonal_scale", layers=layers, dim=dim)
    for layers, dim in [(96, 128), (48, 192), (24, 256), (12, 384), (6, 512), (4, 640)]:
        add_spec(specs, family="rough_iso_budget_ratio", layers=layers, dim=dim)
    return list(specs.values())


def load_specs(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.specs_json:
        loaded = json.loads(pathlib.Path(args.specs_json).read_text())
        if not isinstance(loaded, list):
            raise ValueError("--specs-json must contain a list")
        return loaded
    specs = default_specs()
    if args.limit:
        specs = specs[: args.limit]
    return specs


def build_config(args: argparse.Namespace, spec: Dict[str, object]) -> ModelConfig:
    dim = int(spec["dim"])
    heads = int(spec.get("heads") or choose_heads(dim, args.target_head_dim))
    return ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len,
        n_layer=int(spec["layers"]),
        n_head=heads,
        n_embd=dim,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
    )


def make_run_summary(
    *,
    spec: Dict[str, object],
    cfg: ModelConfig,
    train_records: List[Dict[str, object]],
    eval_records: List[Dict[str, object]],
    estimated_params: Dict[str, int],
    exact_params: int,
    peak_gpu_gb: float | None,
) -> Dict[str, object]:
    predicted = sum(int(row["predicted_tokens"]) for row in train_records)
    elapsed = float(train_records[-1]["clock_seconds"]) if train_records else 0.0
    train_losses = [float(row["loss"]) for row in train_records]
    smoothed = moving_average(train_losses, window=min(100, max(1, len(train_losses))))
    phase_counts: Dict[str, int] = {}
    for row in train_records:
        phase_counts[str(row["phase"])] = phase_counts.get(str(row["phase"]), 0) + 1
    return {
        "name": spec["name"],
        "families": spec["families"],
        "layers": cfg.n_layer,
        "dim": cfg.n_embd,
        "heads": cfg.n_head,
        "head_dim": cfg.head_dim,
        "mlp_ratio": cfg.mlp_ratio,
        "exact_params": int(exact_params),
        "estimated_total_params": estimated_params["total_params"],
        "embedding_params": estimated_params["embedding_params"],
        "nonembed_params": estimated_params["nonembed_params"],
        "embedding_param_fraction": estimated_params["embedding_params"] / estimated_params["total_params"],
        "steps": len(train_records),
        "phase_counts": phase_counts,
        "elapsed_sec": elapsed,
        "predicted_tokens": int(predicted),
        "predicted_tok_per_sec": predicted / elapsed if elapsed > 0 else 0.0,
        "first_train_loss": train_losses[0] if train_losses else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "smoothed_final_train_loss_100": smoothed[-1] if smoothed else None,
        "first_eval_loss": float(eval_records[0]["next_token_eval_loss"]) if eval_records else None,
        "final_eval_loss": float(eval_records[-1]["next_token_eval_loss"]) if eval_records else None,
        "eval_loss_delta": (
            float(eval_records[-1]["next_token_eval_loss"]) - float(eval_records[0]["next_token_eval_loss"])
            if len(eval_records) >= 2
            else None
        ),
        "mean_step_sec": (
            sum(float(row["step_seconds"]) for row in train_records) / len(train_records)
            if train_records
            else None
        ),
        "peak_gpu_gb": peak_gpu_gb,
    }


def run_one(
    *,
    args: argparse.Namespace,
    spec: Dict[str, object],
    index: int,
    out_dir: pathlib.Path,
) -> Dict[str, object]:
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    seed = args.seed + index * 1009
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    cfg = build_config(args, spec)
    estimated = estimate_ngpt_params(
        vocab_size=cfg.vocab_size,
        layers=cfg.n_layer,
        dim=cfg.n_embd,
        mlp_ratio=cfg.mlp_ratio,
    )
    model = EfficientNGPT(cfg).to(device)
    exact_params = exact_param_count(model)
    opt = build_ngpt_adamw(model, lr=args.lr)
    token_path = pathlib.Path(args.tokens)
    tst_batches = TSTBagBatches(
        token_path,
        latent_seq_len=cfg.block_size,
        bag_size=args.bag_size,
        batch_size=args.batch_size,
        device=device,
        split="train",
        seed=seed,
    )
    recovery_batches = make_next_token_batches(
        token_path,
        cfg,
        batch_size=args.batch_size,
        device=device,
        split="train",
        seed=seed + 17,
    )
    eval_batches = make_next_token_batches(
        token_path,
        cfg,
        batch_size=args.eval_batch_size,
        device=device,
        split="val",
        seed=seed + 31,
    )
    train_records: List[Dict[str, object]] = []
    eval_records: List[Dict[str, object]] = [
        {
            "step": -1,
            "clock_seconds": 0.0,
            "phase": "initial",
            "next_token_eval_loss": eval_next_token(
                model,
                eval_batches,
                iters=args.eval_iters,
                dtype=dtype,
                device=device,
            ),
        }
    ]
    start = time.perf_counter()
    transition_sec = args.seconds_per_run * args.superposition_fraction
    next_eval = args.eval_interval_sec
    step = 0
    error_message = None
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= args.seconds_per_run and step > 0:
            break
        phase = "superposition" if elapsed < transition_sec else "recovery"
        if phase == "superposition":
            x, y = tst_batches.next_batch()
        else:
            x, y = recovery_batches.next_batch()
        opt.zero_grad(set_to_none=True)
        synchronize(device)
        step_start = time.perf_counter()
        try:
            with autocast_context(device, dtype):
                if phase == "superposition":
                    _, loss = model.forward_tst_superposition(x, y, bag_size=args.bag_size)
                    predicted_tokens = args.batch_size * cfg.block_size * args.bag_size
                else:
                    _, loss = model(x, y)
                    predicted_tokens = args.batch_size * cfg.block_size
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite loss: {float(loss.detach().cpu())}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
            opt.step()
        except Exception as exc:  # keep partial run evidence on failure
            error_message = repr(exc)
            break
        synchronize(device)
        elapsed = time.perf_counter() - start
        train_records.append(
            {
                "step": step,
                "clock_seconds": elapsed,
                "phase": phase,
                "loss": float(loss.detach().cpu()),
                "grad_norm": grad_norm,
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
                        model,
                        eval_batches,
                        iters=args.eval_iters,
                        dtype=dtype,
                        device=device,
                    ),
                }
            )
            next_eval += args.eval_interval_sec
        step += 1
    if train_records and eval_records[-1]["step"] != train_records[-1]["step"]:
        eval_records.append(
            {
                "step": train_records[-1]["step"],
                "clock_seconds": train_records[-1]["clock_seconds"],
                "phase": train_records[-1]["phase"],
                "next_token_eval_loss": eval_next_token(
                    model,
                    eval_batches,
                    iters=args.eval_iters,
                    dtype=dtype,
                    device=device,
                ),
            }
        )
    peak_gpu_gb = None
    if device.type == "cuda":
        peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9
    summary = make_run_summary(
        spec=spec,
        cfg=cfg,
        train_records=train_records,
        eval_records=eval_records,
        estimated_params=estimated,
        exact_params=exact_params,
        peak_gpu_gb=peak_gpu_gb,
    )
    run = {
        "spec": spec,
        "config": asdict(cfg),
        "args_subset": {
            "bag_size": args.bag_size,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "eval_batch_size": args.eval_batch_size,
            "eval_interval_sec": args.eval_interval_sec,
            "eval_iters": args.eval_iters,
            "lr": args.lr,
            "seconds_per_run": args.seconds_per_run,
            "seq_len": args.seq_len,
            "superposition_fraction": args.superposition_fraction,
        },
        "transition_sec": transition_sec,
        "train": train_records,
        "eval": eval_records,
        "summary": summary,
        "error": error_message,
    }
    (out_dir / f"run_{index:02d}_{spec['name']}.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    del model, opt, tst_batches, recovery_batches, eval_batches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run


def completed_run_files(out_dir: pathlib.Path) -> Dict[str, pathlib.Path]:
    files: Dict[str, pathlib.Path] = {}
    for path in out_dir.glob("run_*.json"):
        try:
            data = json.loads(path.read_text())
            name = str(data["spec"]["name"])
        except Exception:
            continue
        files[name] = path
    return files


def finite_rows(summaries: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for row in summaries:
        loss = row.get("final_eval_loss")
        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            rows.append(row)
    return rows


def ols_fit(rows: Sequence[Dict[str, object]], target: str, *, log_target: bool = False) -> Dict[str, object] | None:
    rows = [r for r in rows if isinstance(r.get(target), (int, float)) and math.isfinite(float(r[target]))]
    if len(rows) < 6:
        return None
    x = []
    y = []
    for r in rows:
        log_p = math.log(float(r["exact_params"]))
        log_l = math.log(float(r["layers"]))
        log_d = math.log(float(r["dim"]))
        log_ratio = math.log(float(r["layers"]) / float(r["dim"]))
        x.append([1.0, log_p, log_l, log_d, log_ratio, log_ratio * log_ratio])
        value = float(r[target])
        y.append(math.log(value) if log_target else value)
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    ridge = 1e-4 * np.eye(X.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + ridge, X.T @ Y)
    pred = X @ beta
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
    return {
        "target": target,
        "features": ["intercept", "log_params", "log_layers", "log_dim", "log_layers_over_dim", "log_ratio_sq"],
        "coef": [float(v) for v in beta],
        "log_target": log_target,
        "rmse": rmse,
        "n": len(rows),
    }


def predict_fit(fit: Dict[str, object], *, params: float, layers: int, dim: int) -> float:
    beta = np.asarray(fit["coef"], dtype=np.float64)
    log_ratio = math.log(layers / dim)
    x = np.asarray(
        [1.0, math.log(params), math.log(layers), math.log(dim), log_ratio, log_ratio * log_ratio],
        dtype=np.float64,
    )
    value = float(x @ beta)
    return math.exp(value) if fit.get("log_target") else value


def candidate_1b_shapes(
    *,
    vocab_size: int,
    mlp_ratio: float,
    target_params: int = 1_000_000_000,
    tolerance: float = 0.04,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for dim in range(512, 4097, 128):
        for layers in range(2, 129):
            est = estimate_ngpt_params(vocab_size=vocab_size, layers=layers, dim=dim, mlp_ratio=mlp_ratio)
            rel_err = abs(est["total_params"] - target_params) / target_params
            if rel_err <= tolerance:
                candidates.append(
                    {
                        "layers": layers,
                        "dim": dim,
                        "heads": choose_heads(dim),
                        "head_dim": dim // choose_heads(dim),
                        "estimated_total_params": est["total_params"],
                        "nonembed_params": est["nonembed_params"],
                        "embedding_param_fraction": est["embedding_params"] / est["total_params"],
                        "relative_param_error": rel_err,
                    }
                )
    return candidates


def analyze(summaries: List[Dict[str, object]], args: argparse.Namespace, out_dir: pathlib.Path) -> Dict[str, object]:
    rows = finite_rows(summaries)
    eval_fit = ols_fit(rows, "final_eval_loss")
    smooth_fit = ols_fit(rows, "smoothed_final_train_loss_100")
    throughput_fit = ols_fit(rows, "predicted_tok_per_sec", log_target=True)
    candidates = candidate_1b_shapes(vocab_size=args.vocab_size, mlp_ratio=args.mlp_ratio)
    projected = []
    for cand in candidates:
        item = dict(cand)
        if eval_fit is not None:
            item["predicted_final_eval_loss"] = predict_fit(
                eval_fit,
                params=float(item["estimated_total_params"]),
                layers=int(item["layers"]),
                dim=int(item["dim"]),
            )
        if smooth_fit is not None:
            item["predicted_smoothed_train_loss"] = predict_fit(
                smooth_fit,
                params=float(item["estimated_total_params"]),
                layers=int(item["layers"]),
                dim=int(item["dim"]),
            )
        if throughput_fit is not None:
            item["predicted_tok_per_sec"] = predict_fit(
                throughput_fit,
                params=float(item["estimated_total_params"]),
                layers=int(item["layers"]),
                dim=int(item["dim"]),
            )
        projected.append(item)
    projected.sort(key=lambda row: row.get("predicted_final_eval_loss", float("inf")))
    analysis = {
        "rows": rows,
        "fits": {
            "final_eval_loss": eval_fit,
            "smoothed_final_train_loss_100": smooth_fit,
            "predicted_tok_per_sec": throughput_fit,
        },
        "one_billion_param_projection": projected[:80] if eval_fit is not None else [],
        "best_projected_1b": projected[0] if projected and eval_fit is not None else None,
        "caveats": [
            "Short-run scaling sweeps estimate early-training efficiency, not converged Chinchilla-optimal loss.",
            "The 1B projection is an extrapolation from the swept sizes; use the next sweep to validate the winning band.",
            "FP8 changes memory/throughput constraints more than the algebraic parameter-count optimum.",
        ],
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    write_markdown_report(analysis, summaries, args, out_dir)
    plot_analysis(analysis, summaries, out_dir)
    return analysis


def family_label(row: Dict[str, object]) -> str:
    families = row.get("families") or []
    if isinstance(families, list):
        return "+".join(str(x) for x in families)
    return str(families)


def plot_analysis(analysis: Dict[str, object], summaries: List[Dict[str, object]], out_dir: pathlib.Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_error.txt").write_text(repr(exc))
        return
    rows = finite_rows(summaries)
    if not rows:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dims = [int(r["dim"]) for r in rows]
    layers = [int(r["layers"]) for r in rows]
    params = [float(r["exact_params"]) for r in rows]
    losses = [float(r["final_eval_loss"]) for r in rows]
    smooth = [float(r["smoothed_final_train_loss_100"]) for r in rows]
    throughput = [float(r["predicted_tok_per_sec"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    sc = ax.scatter(params, losses, c=[math.log(l / d) for l, d in zip(layers, dims)], s=72, cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("final next-token eval loss")
    ax.set_title("Early eval loss vs model size (color = log layers/dim)")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="log(L / d)")
    fig.tight_layout()
    fig.savefig(plot_dir / "eval_loss_vs_params.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    sc = ax.scatter(dims, layers, c=losses, s=90, cmap="magma_r")
    ax.set_xlabel("width d_model")
    ax.set_ylabel("layers")
    ax.set_title("Depth/width map colored by final eval loss")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="final eval loss")
    fig.tight_layout()
    fig.savefig(plot_dir / "depth_width_loss_map.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    ax.scatter(params, throughput, c=dims, s=72, cmap="plasma")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("predicted raw tokens/sec")
    ax.set_title("Measured TST throughput vs size")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "throughput_vs_params.png")

    for family in sorted({fam for row in rows for fam in row.get("families", [])}):
        fam_rows = [r for r in rows if family in r.get("families", [])]
        fam_rows.sort(key=lambda r: (int(r["dim"]), int(r["layers"])))
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
        x = [int(r["dim"]) if "width" in family or "diagonal" in family else int(r["layers"]) for r in fam_rows]
        ax.plot(x, [float(r["final_eval_loss"]) for r in fam_rows], marker="o", label="eval")
        ax.plot(x, [float(r["smoothed_final_train_loss_100"]) for r in fam_rows], marker="o", label="train objective")
        ax.set_xlabel("width d_model" if "width" in family or "diagonal" in family else "layers")
        ax.set_ylabel("loss")
        ax.set_title(f"{family}: loss trend")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{family}_loss_trend.png")

    projected = analysis.get("one_billion_param_projection") or []
    projected = [p for p in projected if "predicted_final_eval_loss" in p]
    if projected:
        fig, ax1 = plt.subplots(figsize=(9.0, 5.4), dpi=150)
        xs = [int(p["dim"]) for p in projected]
        ys = [float(p["predicted_final_eval_loss"]) for p in projected]
        cs = [int(p["layers"]) for p in projected]
        sc = ax1.scatter(xs, ys, c=cs, s=50, cmap="viridis")
        ax1.set_xlabel("1B candidate width d_model")
        ax1.set_ylabel("predicted final eval loss")
        ax1.set_title("Projected 1B iso-param shape curve")
        ax1.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax1, label="layers")
        fig.tight_layout()
        fig.savefig(plot_dir / "projected_1b_shape_curve.png")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), dpi=145)
    axes = axes.ravel()
    axes[0].scatter(params, losses, c=dims, cmap="viridis", s=65)
    axes[0].set_xscale("log")
    axes[0].set_title("eval vs params")
    axes[1].scatter(dims, losses, c=layers, cmap="plasma", s=65)
    axes[1].set_title("eval vs width")
    axes[2].scatter(layers, losses, c=dims, cmap="viridis", s=65)
    axes[2].set_title("eval vs depth")
    axes[3].scatter(smooth, losses, c=throughput, cmap="cividis", s=65)
    axes[3].set_title("eval vs train objective")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "dashboard.png")


def write_markdown_report(
    analysis: Dict[str, object],
    summaries: List[Dict[str, object]],
    args: argparse.Namespace,
    out_dir: pathlib.Path,
) -> None:
    rows = finite_rows(summaries)
    rows_by_eval = sorted(rows, key=lambda r: float(r["final_eval_loss"]))
    best = rows_by_eval[0] if rows_by_eval else None
    projection = analysis.get("best_projected_1b")
    lines = [
        "# enGPT TST Scaling Sweep",
        "",
        f"- Completed runs: {len(rows)}",
        f"- Seconds per run cap: {args.seconds_per_run}",
        f"- Objective: TST superposition for {args.superposition_fraction:.2f} of the run, then next-token recovery.",
        f"- Bag size: {args.bag_size}",
        f"- Sequence length: {args.seq_len}",
        "",
    ]
    if best:
        lines += [
            "## Best Observed Short-Run Shape",
            "",
            (
                f"`{best['name']}`: L={best['layers']}, d={best['dim']}, "
                f"params={int(best['exact_params']):,}, final eval={float(best['final_eval_loss']):.4f}, "
                f"smoothed train={float(best['smoothed_final_train_loss_100']):.4f}, "
                f"throughput={float(best['predicted_tok_per_sec']):,.0f} tok/s."
            ),
            "",
        ]
    if projection:
        lines += [
            "## Best Projected 1B Shape",
            "",
            (
                f"L={projection['layers']}, d={projection['dim']}, heads={projection['heads']}, "
                f"estimated params={int(projection['estimated_total_params']):,}, "
                f"predicted eval={float(projection.get('predicted_final_eval_loss', float('nan'))):.4f}."
            ),
            "",
            "This is a regression projection from short runs, not a claim of converged scaling-law truth until the sweep finishes and the fit residuals are inspected.",
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `analysis.json`: fitted laws and 1B candidates.",
        "- `summary.json`: all run summaries.",
        "- `run_*.json`: raw train/eval curves per config.",
        "- `plots/`: dashboard, axis plots, throughput, and 1B shape projection.",
        "",
        "## Caveats",
        "",
        "- These runs measure early-training efficiency under a fixed time cap.",
        "- TST training objective is cheaper/coarser during phase 1; next-token eval remains the transfer metric.",
        "- FP8 should make wider shapes more attractive operationally, but this script measures BF16 unless launched otherwise.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def write_summary(
    *,
    args: argparse.Namespace,
    specs: List[Dict[str, object]],
    runs: List[Dict[str, object]],
    out_dir: pathlib.Path,
) -> List[Dict[str, object]]:
    summaries = [run["summary"] for run in runs]
    payload = {
        "args": vars(args),
        "specs": specs,
        "summaries": summaries,
        "completed": len(runs),
        "planned": len(specs),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return summaries


def print_specs(specs: List[Dict[str, object]], args: argparse.Namespace) -> None:
    rows = []
    for spec in specs:
        est = estimate_ngpt_params(
            vocab_size=args.vocab_size,
            layers=int(spec["layers"]),
            dim=int(spec["dim"]),
            mlp_ratio=args.mlp_ratio,
        )
        rows.append(
            {
                "name": spec["name"],
                "families": spec["families"],
                "layers": spec["layers"],
                "dim": spec["dim"],
                "heads": choose_heads(int(spec["dim"]), args.target_head_dim),
                "estimated_params": est["total_params"],
                "nonembed_params": est["nonembed_params"],
            }
        )
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out-dir", default="runs/scaling_sweep")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seconds-per-run", type=float, default=300.0)
    parser.add_argument("--superposition-fraction", type=float, default=0.4)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--eval-interval-sec", type=float, default=90.0)
    parser.add_argument("--eval-iters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--specs-json", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            # The H100 test box has a CUDA/NVRTC mix that can advertise cuDNN
            # SDPA engines and then fail at runtime. Flash/math SDPA are stable.
            torch.backends.cuda.enable_cudnn_sdp(False)
    specs = load_specs(args)
    if args.dry_run:
        print_specs(specs, args)
        return
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "launch_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    done = completed_run_files(out_dir) if args.resume else {}
    runs: List[Dict[str, object]] = []
    for path in sorted(done.values()):
        runs.append(json.loads(path.read_text()))
    for index, spec in enumerate(specs):
        if args.resume and spec["name"] in done:
            print(f"[skip] {spec['name']} already complete")
            continue
        print(f"[run {index + 1}/{len(specs)}] {spec['name']} families={','.join(spec['families'])}", flush=True)
        run = run_one(args=args, spec=spec, index=index, out_dir=out_dir)
        runs.append(run)
        summaries = write_summary(args=args, specs=specs, runs=runs, out_dir=out_dir)
        analyze(summaries, args, out_dir)
        if run.get("error"):
            print(f"[error] {spec['name']}: {run['error']}", flush=True)
        else:
            summary = run["summary"]
            print(
                f"[done] {spec['name']} eval={summary['final_eval_loss']:.4f} "
                f"train={summary['smoothed_final_train_loss_100']:.4f} "
                f"tok/s={summary['predicted_tok_per_sec']:.0f}",
                flush=True,
            )
    summaries = write_summary(args=args, specs=specs, runs=runs, out_dir=out_dir)
    analysis = analyze(summaries, args, out_dir)
    print(json.dumps({"completed": len(runs), "best_projected_1b": analysis.get("best_projected_1b")}, indent=2))


if __name__ == "__main__":
    main()
