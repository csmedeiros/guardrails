# Architecture and data flow

```
hf dataset ──► prep.py ──► data/processed/stage1_{en,pt}.parquet ─┐
                                                                    ├─► embed.py (frozen BGE-M3) ─► embeddings/*.npz
vLLM (Qwen3-4B-AWQ :8011) ─► gen_synth.py (Data Designer) ─► data/synthetic/synth_v1.parquet
                              └► prep_synth.py ─► data/processed/synth.parquet ─┘
embeddings ─► train_head.py / baseline_torch.py (stage 1 experiments)
embeddings ─► stage2.py (stage 1 → stage 2 fine-tune) ─► models/head_*.npz ─► models/head.npz
models/head.npz + BGE-M3 ─► guard_model.GuardModel ─► serve.py (FastAPI :8100) ◄─ guardrails_config/actions.py ◄─ NeMo Guardrails
every training script ─► tracking.start() ─► mlflow/mlflow.db (+ mlruns/ artifacts)
```

## Design decisions (and why)

- **Frozen multilingual encoder + small head.** BERTimbau is Portuguese-only; EN, PT-BR and ES are all required. Embeddings are
  computed once and cached, so heads train in seconds and experiments are cheap.
- **Binary safe/unsafe, recall-first.** A missed unsafe prompt costs more than a blocked safe one: the threshold is the highest
  value reaching 95% unsafe recall on validation; ROC AUC and recall@FPR are the headline ranking metrics.
- **Group split by prompt.** An English prompt and its Portuguese translation always land in the same split; otherwise test is leaked.
  Ties in the vote go to `unsafe`.
- **Replay in stage 2.** Fine-tuning only on synthetic data forgets real data (−5.5 points on BeaverTails), so a share of stage-1
  training data is mixed in.
- **Weights as plain `.npz` arrays**, no pickle: inference is a numpy forward pass after the encoder.
- **Two Python envs.** vLLM and data-designer pin conflicting `pyarrow`/Python ranges; vLLM lives in `.venv-vllm`.

## Files

| File | Role |
|---|---|
| `prep.py` | keep EN/PT-agreeing rows, group by prompt, 70/15/15 split (seed 42) |
| `embed.py` | `embed.py <parquet> <out.npz>`: BGE-M3, max_len 256, L2-normalised, stored as float16 |
| `train_head.py` | sklearn logreg/MLP head experiments (`<head> <run_name> <npz...>`) |
| `baseline_torch.py` | PyTorch MLP with per-epoch curves and weight plots (`<run_name> <npz...>`) |
| `gen_synth.py` | Data Designer pipeline: label × language × style × topic → text → independent judge verdict |
| `prep_synth.py` | keep rows where judge verdict == label, dedupe, split |
| `stage2.py` | `stage2.py <replay> <tag>`: stage 1 then warm-start fine-tune, thresholds, MLflow, saves head |
| `prep_ext.py` | PolyGuardPrompts EN/PT/ES + Nemotron-v3 ES -> `data/processed/{pgp_*,nem_es}.parquet` (adds `adv` column) |
| `stage3.py` | `stage3.py <tag> <replay> <syn,nem> [target_recall]`: stage-3 fine-tune incl. the Spanish train split |
| `eval_heads.py` | evaluate any `head*.npz` on every held-out set at its own frozen threshold; MLflow `10-eval-heads-all-benchmarks` |
| `guard_model.py` | `GuardModel`: encoder + numpy head; env `GUARD_HEAD`, `GUARD_ENCODER`, `GUARD_DEVICE` |
| `serve.py` | FastAPI app, `/health`, `/classify`, `/classify_batch` |
| `guardrails_config/` | `config.yml` (input flow), `rails.co` (Colang), `actions.py` (calls `GUARD_URL`) |

## Data file schemas

- `stage1_*.parquet`: `text, split, unsafe (0/1), lang`.
- `synth.parquet`: same columns. `synth_v1.parquet` (raw): `label, language, style, topic, text, verdict`.
- `embeddings/*.npz`: `x` float16 [n,1024], `y` int8, `split` U8, `lang` U4 (load with `allow_pickle=False`).
- `models/head.npz`: `w0,b0,w1,b1,w2,b2`, `thr`.
