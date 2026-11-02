# Reproducing the experiments

All commands run from the repository root. `make -n <target>` prints what a target does.

## Requirements

| Need | For | Notes |
|---|---|---|
| [uv](https://docs.astral.sh/uv/), Python ≥3.12 | everything | `make setup` creates `.venv` (tested on 3.13) |
| ~10 GB disk | main env + model cache | `.venv-vllm` adds ~9 GB |
| NVIDIA GPU | embedding (optional), synthetic generation (required) | tested on RTX 2060, 6 GB, Turing |
| Internet | Hugging Face downloads | dataset (9 MB), `BAAI/bge-m3` (~2.3 GB), `Qwen/Qwen3-4B-AWQ` (~2.5 GB, generation only) |

Without a GPU you can still do everything except `make gen`; embedding ~30k short prompts on CPU takes much longer.

## Path A: reproduce from the shipped synthetic set (recommended)

```bash
make setup
make data          # data/raw/guardbench-ptbr/guardbench_ptbr.jsonl
make prep          # data/processed/stage1_{en,pt}.parquet, stage1_stats.json
make embed         # embeddings/stage1_{en,pt}_bgem3.npz  (frozen BGE-M3, max_len 256)
make baseline      # MLflow run baseline_torch_mlp_detailed (experiment stage1)
make stage2        # prep-synth + embed-synth + stage 1 -> stage 2, replay 1.0, writes models/head.npz
```

`make stage2` runs `prep_synth.py` on the committed `data/synthetic/synth_v1.parquet`; the judge-agreement filter and the
70/15/15 split are seeded (42), so the splits are identical on every machine.

Compare replay shares: `PYTHONPATH=src uv run python src/guardrails/training/stage2.py 0.0 final`, then `0.3`, then `1.0`
(each writes `models/head_final_replay<r>.npz`).

### Expected numbers

Exact values can move in the last decimal with GPU, library or driver differences (fp16 embeddings, sklearn). If you are more
than ~1 point off, something differs; see Troubleshooting.

| Check | Expected |
|---|---|
| `stage1_stats.json` | rows 16,188; agree 14,968; unique prompts 14,922; unsafe rate 0.536; splits 10,444 / 2,239 / 2,239 |
| `synth_stats.json` | generated 6,000; kept 4,510; unsafe rate 0.412; splits 3,164 / 666 / 680; en 1,414 / pt 1,562 / es 1,534 |
| Baseline torch MLP (stage 1 only) | best epoch ≈5; test acc 84.7% at the recall threshold (86.3% at 0.5); AUC 0.940; recall 94.2% |
| Stage 2, replay 1.0 | BeaverTails test acc 87.2%, AUC 0.949, recall 93.1%; synthetic test acc 97.65%, AUC 0.998 |
| Stage 2, replay 0.0 / 0.3 | BeaverTails test acc 81.7% / 86.5% |

Metrics are in MLflow experiments `stage1` and `stage2`.

## Independent evaluation and stage 3 (Spanish train split)

```bash
make data-ext      # PolyGuardPrompts (51 MB) + Nemotron-v3 es/* (42 MB)
make embed-ext     # prep_ext.py then embeddings/{pgp_en,pgp_pt,pgp_es,nem_es}_bgem3.npz
make stage3        # needs the stage-2 head: trains the syn_only (90/85), es_nem and es_syn_nem (95/90/85) heads
make eval-ext      # table of all heads on every held-out set -> reports/external_eval.json + MLflow `10-eval-heads-all-benchmarks`
```

Expected (default head, accuracy): PGP EN 69.8 / PT 68.2 / ES 68.9, Nemotron ES 74.0; `head_es` (Syn+Nem, target 0.90): BT 86.2,
PGP ES 68.6, Nemotron ES 78.6. `ext_stats.json`: Nemotron ES train 7,920 / val 1,922 / test 2,705; PGP 1,725 rows per language.
Stage 3 takes ~4-5 min per head on CPU (sklearn). Run `stage3.py <tag> <replay> <syn|nem|syn,nem> [target]` for other variants.

## Path B: regenerate the synthetic data (NVIDIA GPU)

This yields a *different* dataset (LLM sampling is not bit-reproducible), so numbers will differ slightly from the paper.

```bash
make setup-vllm    # separate .venv-vllm (Python 3.12, vllm 0.33.0)
make vllm-up       # terminal 1: serves Qwen/Qwen3-4B-AWQ as model "gen" on :8011
make gen           # terminal 2: 6,000 records via NeMo Data Designer, ~5-7 records/s (~15-20 min)
make stage2
```

vLLM flags matter on a 6 GB Turing card: `--dtype half` (no bf16), `--max-model-len 4096`, `--gpu-memory-utilization 0.85`,
`--enforce-eager`. FlashAttention-2 is unavailable on Turing; vLLM falls back to the Triton backend automatically. Stop the
vLLM server before embedding so the GPU is free. Smoke-test generation first with
`PYTHONPATH=src uv run python src/guardrails/synth/gen_synth.py 20 preview`.

## Serve and integrate

```bash
make serve                          # GUARD_HEAD, GUARD_ENCODER, GUARD_DEVICE (cpu|cuda) override defaults
make demo                           # expects 3 BLOCKED, 3 allowed
GUARD_URL=http://host:8100 make demo   # rail pointing at another host
```

## MLflow

Runs are stored in `mlflow/mlflow.db` (SQLite). `make mlflow-ui` serves them on http://127.0.0.1:5000. Experiments created
without an explicit location store artifacts in `mlruns/`, not `mlflow/artifacts/`; the UI still serves them as long as it is
started from the repository root.

## Paper

`make paper` needs [tectonic](https://tectonic-typesetting.github.io/). Figures in `paper/figs/` were exported from the
`baseline_torch_mlp_detailed` run (`plots/loss`, `recall`, `weight_distributions`) and the stage-2 confusion matrix.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Address already in use` on vLLM | pick another port and change `ENDPOINT` in `src/guardrails/synth/gen_synth.py` (it is `:8011`) |
| CUDA OOM during embedding | stop vLLM first; lower `bs` in `embed.py` |
| `data-designer` / `pyarrow` pin conflict | keep data-designer in the main env, vLLM in `.venv-vllm`; do not merge them |
| sklearn `NoneType - float` in stage 2 | already handled by resetting `best_loss_`; do not remove that line in `stage2.py` |
| `ModuleNotFoundError: guardrails` | run with `PYTHONPATH=src` (the Makefile does) |
| Numbers differ by several points | check `data/processed/*_stats.json` against the table above first; split changes explain most drift |
