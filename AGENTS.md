# AGENTS.md

Multilingual (EN/PT-BR/ES) binary guardrail classifier: frozen BAAI/bge-m3 + MLP head, trained in two stages
(BeaverTails EN/PT, then synthetic data from NeMo Data Designer + local vLLM), served with FastAPI and as a NeMo Guardrails
input rail. Read `README.md` first; details in `docs/`.

## Ground rules

- **Evidence over assumption.** Answer from files, MLflow, or a command you ran. Numbers come from `docs/RESULTS.md`, the paper
  (`paper/main.tex`) or MLflow, never from memory. If you cannot verify, say what you tried.
- **Keep the claims honest.** Never describe the BeaverTails score as the human-labelled benchmark, never quote the 99.6% synthetic
  Spanish score as Spanish quality (independent Spanish accuracy is 68.9% PolyGuardPrompts / 74.0% Nemotron-v3), never present
  synthetic accuracy as real-world accuracy. See "Known limits" below.
- **Do not edit the selected model or the shipped data casually.** `models/head.npz`, `models/head_es.npz` and `data/synthetic/synth_v1.parquet`
  back the paper's numbers. Regenerating them changes results; if you do, update `docs/RESULTS.md`, `README.md` and the paper.
- **Never commit** `data/raw/`, `data/processed/` (derived from a CC BY-NC dataset), `embeddings/`, `mlflow/`, `mlruns/`,
  `.venv*/`, `logs_*.txt`. `.gitignore` already blocks them.
- **Never merge or push on the user's behalf** unless asked.
- **Present the test plan and get approval before running any evaluation, benchmark or test.** Before the first run, state in
  plain language: which models, which dataset/split, which metric and judge, what it costs (API tokens, GPU, time), and which
  models or runs you will leave out. Wait for the user's yes. A request like "compare with X" or "run the benchmark" is not
  approval of the plan you choose. The approval covers only what was presented: adding a model, dataset, judge or repeating a
  run needs a new approval. If the user narrows the plan, stop the extra runs at once and report what was stopped.
- **Published model:** `caiosmedeiros/prompt-safety-v1` on Hugging Face (= `models/guard_v3_tuned_A`, CC BY-NC 4.0). The training pipeline stays in the repo; the HF download is only a shortcut for using the model. Never log prompt text in the API.
- Helper scripts for tooling/automation are Node.js; the ML stack stays Python.

## Commands (repo root; `make -n <t>` to preview)

| Goal | Command |
|---|---|
| install | `make setup` (uv, `.venv`) |
| get source data | `make data` |
| stage-1 prep / embed / baseline | `make prep`, `make embed`, `make baseline` |
| stage 1 + 2 from shipped synthetic set | `make stage2` (replay via `SYN_REPLAY=0.3 make stage2`) |
| regenerate synthetic data (GPU) | `make setup-vllm`, `make vllm-up` (blocks), then `make gen` |
| independent eval + Spanish train split | `make data-ext embed-ext`, `make stage3 eval-ext` |
| serve API / NeMo demo | `make serve` (:8100), `make demo` |
| pick the served guard model | `GUARD_MODEL=<name> make serve`; names in `configs/guard_models.yaml` (default `prompt_safety_v1`, downloaded from Hugging Face on first use) |
| MLflow UI | `make mlflow-ui` (:5000) |
| paper | `make paper` (tectonic) |

Python modules import `guardrails.*`, so run scripts with `PYTHONPATH=src` (the Makefile does). There is no test suite; the
verification is reproducing the numbers in `docs/REPRODUCE.md` and the 3-blocked/3-allowed `make demo` check.

## Layout

`src/guardrails/` pipeline code, split by job: `core/` (GuardNet, GuardModel, int8 encoder), `serving/` (FastAPI), `data/` (prep, embed),
`synth/` (Data Designer generators, Jev judge), `training/` (heads, stage 2/3, v3 pairs, fine-tuning, Optuna), `evaluation/` (eval_heads, eval_e2e),
`core/registry.py` + `configs/guard_models.yaml` (guard model selection; third-party guards in `models/third_party/`, git-ignored), `tracking.py` (MLflow entry point shared by all) · `guardrails_config/` NeMo rail · `models/head.npz` selected head · `data/synthetic/` shipped
synthetic set · `paper/` IEEE LaTeX · `docs/` ARCHITECTURE, REPRODUCE, RESULTS · `notes/findings.md` research notes ·
`.claude/skills/` agent skills (`reproduce`, `ask-guardrails`).

## Gotchas (each cost real time once)

- vLLM and data-designer live in **separate envs** (`.venv-vllm` vs `.venv`) because of dependency pins. Do not install one into the other.
- vLLM on a 6 GB Turing GPU needs `--dtype half --enforce-eager --max-model-len 4096`; port 8000 may be taken, we use **8011**.
  Stop vLLM before embedding or the GPU runs out of memory.
- `stage2.py` resets `clf.best_loss_` after early stopping; removing that line breaks warm-start fine-tuning.
- Load `.npz` files without `allow_pickle`; strings are fixed-width unicode arrays on purpose.
- Splits are grouped by prompt (an EN prompt and its PT translation must share a split). Do not split by row.
- MLflow runs live in `mlflow/mlflow.db`, but artifacts for experiments created by default are in `mlruns/`.

## Known limits (state these when reporting results)

BeaverTails labels = 5-LLM vote, no humans · independent accuracy is only ~69-74% (PolyGuardPrompts EN/PT/ES, Nemotron-v3 ES), same in
all languages, worst on adversarial prompts (49.6% recall) · Spanish benchmarks are machine-translated/LLM-adapted · synthetic accuracy
optimistic (same model family generates and judges) · poor calibration (threshold 0.017, BeaverTails FPR 19.8%) · input guard only, no dedicated adversarial/obfuscation test beyond the benchmark subsets ·
stage-1 data is CC BY-NC 4.0 · no `LICENSE` file chosen yet.
