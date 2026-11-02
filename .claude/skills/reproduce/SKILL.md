---
name: reproduce
description: Reproduce the guardrail classifier experiments on this machine: set up the environment, download the source data, rebuild splits/embeddings, run stage 1 and stage 2 training, optionally regenerate the synthetic dataset with vLLM, and check the results against the expected numbers. Use when asked to reproduce, rebuild, retrain, rerun experiments, or verify the repo works.
---

# Reproduce the experiments

Source of truth: `docs/REPRODUCE.md` (commands, expected numbers, troubleshooting). Follow it; this skill is the checklist.

## 1. Pick the path (ask the user only if unclear)

- **A. From the shipped synthetic set** (default): no vLLM, no generation.
- **B. Regenerate synthetic data**: needs an NVIDIA GPU with ≥6 GB; results will differ slightly. Only if the user asks.

## 2. Preflight (run, don't assume)

```bash
uv --version; nvidia-smi --query-gpu=name,memory.free --format=csv,noheader; df -h . | tail -1
test -f data/synthetic/synth_v1.parquet && echo shipped-synth-ok
```

No GPU is fine for path A (embedding falls back to CPU, slower). Need roughly 10 GB free (20 GB for path B).

## 3. Path A

```bash
make setup
make data
make prep
make embed
make baseline        # optional, detailed stage-1 curves in MLflow
make stage2          # SYN_REPLAY=1.0 by default; writes models/head.npz
```

Each step prints or writes stats: check `data/processed/stage1_stats.json` and `synth_stats.json` against the expected table
in `docs/REPRODUCE.md` **before** training; a mismatch there explains any later drift.

## 4. Path B extras

```bash
make setup-vllm
make vllm-up         # background; wait for GET http://localhost:8011/v1/models to answer
PYTHONPATH=src uv run python src/guardrails/synth/gen_synth.py 20 preview   # eyeball a few rows per language first
make gen             # overwrites data/synthetic/synth_v1.parquet (back it up if you want the shipped one)
# stop vLLM, then:
make stage2
```

## Optional: independent evaluation + stage 3

`make data-ext embed-ext stage3 eval-ext` (see `docs/REPRODUCE.md`); compare with the expected numbers listed there.

## 5. Verify

1. Read the metrics from MLflow (`make mlflow-ui`, experiment `stage2`) or from the `stage2.py` output.
2. Compare to `docs/REPRODUCE.md` "Expected numbers". Within ~1 point is a pass for path A; path B is expected to differ.
3. Serving check: `make serve` in the background, `curl localhost:8100/health`, then `make demo` must show 3 BLOCKED and 3 allowed.
4. Stop every background server you started.

## 6. Report

State which path, the commands run, the measured numbers next to the expected ones, and any deviation with its cause. If a
step failed, give the real error output and the troubleshooting row you tried. Do not claim reproduction from a partial run.
