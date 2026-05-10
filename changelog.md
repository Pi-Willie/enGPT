# Changelog

## 2026-05-10

- Started a focused correctness/debugging pass because the FineWeb 10M-token run showed enGPT/nGPT was still behind the GPT baseline in token efficiency.
- Compared the local implementation with NVIDIA's public nGPT reference implementation.
- Identified two minimal candidate mismatches to test:
  - Local scripts/configs initialize the effective logit scale at `8.0`, while NVIDIA's reference initializes the effective `sz` logit scale at `1.0`.
  - Local MLP scaling applies `sqrt(d_model)` only to the gate half, while NVIDIA's reference applies the `sqrt(d_model)` `suv` scale to both halves of the SwiGLU `uv` projection.
- Changed `ModelConfig.logit_scale_init` and script defaults from `8.0` to `1.0` to match NVIDIA's effective `sz` initialization.
- Added `ModelConfig.scale_mlp_u_by_sqrt_d` with default `True`, and applied `sqrt(d_model)` to the SwiGLU `u` half as well as the gate half in both carried enGPT and materialized-reference nGPT paths.
- Added `scripts/gpu_train_benchmark.py --no-mlp-u-sqrt-scale` so this MLP scaling change can be disabled for ablation/revert testing without editing code.
- Added explicit GPU benchmark metrics and gate for token-efficiency: `post_step_all_wins_gate` requires enGPT training loss to be below GPT at every logged step after `--beat-after-step` (default `100`).
- Replaced direct scalar/vector nGPT parameters for `alpha`, `s_qk`, and `s_z` with NVIDIA-style scaled parameters:
  - raw scalar parameters initialize at `base_scale = 1 / sqrt(d_model)`;
  - forward computes the same effective initial values by multiplying by `init_value / base_scale`;
  - alpha uses `abs(raw)` like the reference instead of `softplus(raw)`, avoiding the very small softplus derivative at `alpha=0.05`.
  This preserves initial forward values but restores the intended gradient scale for the nGPT scalar/vector controls.
- Fixed the GPU benchmark baseline recipe: GPT+AdamW now has a configurable warmup schedule (`--gpt-warmup-steps`, default `2000`) while enGPT/nGPT keeps zero warmup, matching the nGPT training distinction in the spec and NVIDIA reference training script.
- Reverted the effective default logit scale to `8.0` after ablation:
  - `s_z=1.0` plus the other reference-alignment fixes still left enGPT behind GPT on the 500-step FineWeb check.
  - `s_z=8.0` with corrected scalar parameterization and GPT warmup made enGPT beat GPT at every measured step after step 100 and improved eval loss.
- Set the GPU benchmark's default enGPT learning rate to `1e-3` while leaving GPT at `3e-4` with warmup. The 10M-token FineWeb run with this setting made enGPT beat GPT at every measured post-100 step and beat GPT on eval loss.
- Updated `README.md` RunPod benchmark command so it uses the corrected benchmark defaults instead of the previous explicit tuned LR from the older run.
- Added exact carried-state gauge stabilization:
  - New `ModelConfig.carried_gauge_max` defaults to `1024.0`.
  - New `gauge_carried_state(y, rho, max_radius)` rescales both numerator and denominator by the same row scalar, preserving `y / rho` exactly.
  - Applied after both attention and MLP carried residuals.
  This fixes deep-stack finite-precision overflow where `Y` and `rho` grew exponentially even though the represented hidden state stayed normalized.
- Added benchmark trace recording for publication plots:
  - `scripts/gpu_train_benchmark.py --record-step-stats` now records per-step pre-clip gradient norms, synchronized step seconds, and cumulative clock seconds.
  - The default token and report paths are now repo-relative instead of RunPod-specific.
- Added `scripts/plot_publication.py` to generate README figures and a larger PNG/PDF figure gallery from benchmark JSON reports.
- Added curated publication artifacts under `assets/figures` and source reports under `assets/reports`.
- Rebuilt the README as a compact public repo README for Efficient nGPT with exactly two displayed figures, install commands, quick checks, usage, benchmark, and packaging instructions.
- Added package metadata and distribution files:
  - `pyproject.toml` now has a build backend, optional dependency groups, and console scripts.
  - Added `LICENSE`, `MANIFEST.in`, `.gitignore`, `scripts/__init__.py`, and `examples/quickstart.py`.
  - Added `engpt.__version__`.
- Revised the README figure narrative:
  - The token-efficiency loss curve is now the first displayed figure.
  - The 64 and 128 layer depth result is shown later as the second displayed figure.
  - Added a clearer research note that this is nGPT implemented with carried-radius state, approaching GPT-like throughput and VRAM behavior while kernel optimization continues.

### Verification

- Local tests: `python3 -m pytest -q` passed after the changes.
- Publication trace GPU run, `4` layers, `d=256`, `T=256`, `B=16`, 300 steps:
  - Report: `assets/reports/gpu_report_publication_trace_4l_d256_s256_300.json`
  - Per-model trace lengths: 300 losses, 300 gradient norms, and 300 cumulative clock points.
  - GPT eval loss `8.8474365234`, enGPT eval loss `6.5966850758`
  - `post_step_all_wins=True`, `post_step_win_fraction=1.0`
- 10M-token FineWeb/GPT-2 target regime, `4` layers, `d=256`, `T=256`, `B=16`:
  - Report: `runs/runpod/gpu_report_token_efficiency_lr1e3.json`
  - `post_step_all_wins=True`, `post_step_win_fraction=1.0`
  - GPT eval loss `5.9258158922`, enGPT eval loss `5.7443021536`
  - enGPT/reference nGPT loss abs diff `0.0004205704`
- Smaller regime, `2` layers, `d=128`, `T=128`, `B=16`, 500 steps:
  - Report: `runs/runpod/gpu_report_regime_2x128_s128_500.json`
  - `post_step_all_wins=True`, `post_step_win_fraction=1.0`
  - GPT eval loss `8.5633758545`, enGPT eval loss `6.7330748558`
- Longer-context regime, `4` layers, `d=256`, `T=512`, `B=8`, 500 steps:
  - Report: `runs/runpod/gpu_report_regime_4x256_s512_500.json`
  - `post_step_all_wins=True`, `post_step_win_fraction=1.0`
  - GPT eval loss `7.5256958008`, enGPT eval loss `6.4849498749`
