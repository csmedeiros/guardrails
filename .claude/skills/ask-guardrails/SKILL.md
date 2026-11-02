---
name: ask-guardrails
description: Answer questions about this guardrail classifier project (how it works, data, training stages, metrics, thresholds, API, NeMo Guardrails integration, limitations, licences) using evidence from the repository. Use for any "how/why/what/does it..." question about the repo, results or model behaviour.
---

# Answer questions about this project

Answer from evidence in the repo or a command you ran; never from memory. Quote the file and line or the number's source.

## Where answers live

| Question about | Read |
|---|---|
| what it is, quick start | `README.md` |
| pipeline, files, design reasons | `docs/ARCHITECTURE.md` |
| commands, expected numbers, errors | `docs/REPRODUCE.md` |
| metrics, ablations, caveats | `docs/RESULTS.md`, `paper/main.tex` |
| exact behaviour (splits, thresholds, training) | `src/guardrails/*.py` (`prep.py`, `stage2.py`, `guard_model.py`, `serve.py`) |
| NeMo Guardrails rail | `guardrails_config/{config.yml,rails.co,actions.py}` |
| dataset licences, hardware, vLLM | `notes/findings.md` |
| a specific run | MLflow: `PYTHONPATH=src uv run python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow/mlflow.db'); print(mlflow.search_runs(search_all_experiments=True).T)"` (needs the runs to exist locally) |

## Method

1. Find the files above that bear on the question; read them (the code is short).
2. If the answer is a behaviour ("does the endpoint embed text?", "is the split grouped?"), confirm in the code, and run it when it is cheap (`make serve` + `curl`, a few lines of Python on `models/head.npz`).
3. If a number is asked, take it from `docs/RESULTS.md` or MLflow and say which test set it is on and the threshold.
4. Say plainly when the repo does not answer it, with what you looked at.

## Facts you must not distort

- BeaverTails (`marcusnogueira/guardbench-ptbr`) labels come from a 5-LLM majority vote, prompts only, no human labels, CC BY-NC 4.0.
- 87.2% is accuracy on the held-out BeaverTails test; 97.65% is on held-out synthetic data and is optimistic.
- 99.6% on Spanish is synthetic and in-distribution. Independent Spanish accuracy: 68.9% (PolyGuardPrompts) and 74.0% (Nemotron-v3
  test). English and Portuguese rows of PolyGuardPrompts score 69.8% and 68.2%, so the gap is prompt style (adversarial prompts), not language.
- Training on the Nemotron-v3 Spanish train split (`head_es.npz`) raises Nemotron-ES accuracy to 78.6% but barely changes PolyGuardPrompts;
  the default head stays `head.npz`.
- The threshold (0.0174) is chosen on validation for 95% unsafe recall; calibration is poor.
- It is an input guard (prompts), not a response moderator. Adversarial robustness was only measured on the adversarial subsets of PolyGuardPrompts/Nemotron-v3 (poor: 49.6% recall on adversarial Spanish).
- The API embeds raw text itself; clients send text, not vectors.
