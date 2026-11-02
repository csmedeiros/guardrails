# Results and how to read them

Selected model: stage 1 → stage 2 with replay 1.0 (`models/head.npz`, threshold 0.0174). Threshold = highest value reaching
95% unsafe recall on the pooled validation sets, frozen before test.

## Test metrics

| Set | Acc | ROC AUC | Unsafe recall | FPR |
|---|---|---|---|---|
| BeaverTails test | 87.2% | 0.949 | 93.1% | 19.8% (recall@FPR5% = 74.7%) |
| Synthetic test | 97.65% | 0.998 | 99.6% | 3.7% |

Per language: BeaverTails EN 86.5 / PT 87.9; synthetic EN 97.1 / PT 96.2 / ES 99.6.
Stage-1-only head on the synthetic test: 83.2% accuracy (shows the gain from stage 2).

## Independent evaluation (data the model never saw, other sources)

Default head (`models/head.npz`), frozen threshold 0.0174. PolyGuardPrompts (PGP, CC BY 4.0, machine-translated, test split)
and Nemotron-Safety-Guard-Dataset-v3 Spanish (CC BY 4.0). Prompts only. Reproduce: `make data-ext embed-ext stage3 eval-ext`
(writes `reports/external_eval.json`, MLflow experiment `10-eval-heads-all-benchmarks`).

| Set | Subset | n | Acc | Unsafe recall | FPR | AUC |
|---|---|---|---|---|---|---|
| PGP | EN | 1725 | 69.8% | 69.6% | 30.1% | 0.773 |
| PGP | PT-BR | 1725 | 68.2% | 64.4% | 28.8% | 0.749 |
| PGP | ES | 1725 | 68.9% | 63.3% | 26.9% | 0.750 |
| PGP | ES non-adversarial | 915 | 73.8% | 75.1% | 27.2% | 0.814 |
| PGP | ES adversarial | 810 | 63.3% | 49.6% | 26.6% | 0.660 |
| Nemotron-v3 | ES test | 2705 | 74.0% | 74.0% | 26.0% | 0.819 |
| Nemotron-v3 | ES jailbreak | 791 | 73.7% | 81.5% | 38.3% | 0.813 |

- **Language vs prompt style:** EN, PT and ES rows of the same dataset are within two points, so the drop is not about Spanish. The
  weakness is adversarial prompts (jailbreak, role-play, hypothetical framing). Median prompt length is similar (~260 chars).
  Machine-translation noise was not measured.
- No PGP English prompt is in BeaverTails or the synthetic set (exact match: 0 of 1,725). One Nemotron test prompt also appears in
  its train split after case normalisation.

## Stage 3: training on the Spanish train split (Nemotron-v3 `es/train`, 7,920 prompts, capped at 8,000 before dedup)

Accuracy (%) at each head's own validation-chosen threshold; "target" = the unsafe recall used to choose it on pooled validation.

| Head | Target | BT test | Syn test | PGP EN | PGP PT | PGP ES | Nem ES |
|---|---|---|---|---|---|---|---|
| Syn only (default `head.npz`) | 95 | 87.2 | 97.6 | 69.8 | 68.2 | 68.9 | 74.0 |
| Syn only | 90 | 87.9 | 97.8 | 71.4 | 70.1 | 69.5 | 73.2 |
| Syn only | 85 | 85.8 | 97.1 | 70.1 | 68.5 | 69.4 | 70.8 |
| Syn + Nem | 95 | 83.3 | 96.3 | 66.4 | 65.5 | 63.1 | 74.9 |
| **Syn + Nem (`head_es.npz`)** | 90 | 86.2 | 96.9 | 71.0 | 70.1 | 68.6 | 78.6 |
| Syn + Nem | 85 | 86.6 | 97.1 | 72.3 | 71.2 | 69.8 | 78.6 |
| Nem only | 95 | 83.0 | 81.6 | 64.9 | 64.8 | 61.0 | 73.1 |

AUC (threshold-free): Nemotron ES 0.819 → 0.864; PGP ES 0.750 → 0.768, PGP EN 0.773 → 0.790, PGP PT 0.749 → 0.769; BeaverTails 0.949 → 0.938.

- Adding Spanish data helps its own distribution (+4.6 points at a comparable operating point) and transfers little (≤ about 3 points on PGP).
- At the original 95% recall target the Spanish data push the threshold to 0.0003 and the false-positive rate to 46-56%, so accuracy
  falls. Compare heads at similar recall/FPR, not at equal targets.
- Even in-distribution AUC is only 0.864: likely the ceiling of a frozen encoder + small head, or label noise (Nemotron labels
  partly come from an LLM jury). Unfreezing the encoder (LoRA) was not tried.
- Default stays `head.npz`; `head_es.npz` is optional.

## Replay ablation (Accuracy at the validation-chosen threshold)

| Replay | Val pooled | BT test | BT AUC | Syn test | Syn AUC |
|---|---|---|---|---|---|
| 0.0 | 85.5 | 81.7 | 0.921 | 98.2 | 0.999 |
| 0.3 | 88.9 | 86.5 | 0.947 | 97.8 | 0.998 |
| 1.0 | 89.1 | 87.2 | 0.949 | 97.7 | 0.998 |

## Caveats that change how you should use the numbers

- **BeaverTails here is not the human-labelled benchmark.** `guardbench-ptbr` holds prompts only, labels are a five-LLM
  majority vote, and rows where the English and Portuguese labels disagree (1,220) were dropped.
- **Spanish synthetic 99.6% is in-distribution.** On independent Spanish data the model reaches 68.9% (PGP) and 74.0% (Nemotron-v3).
  Both are machine-translated or LLM-adapted, not native human-written. GuardBench's Spanish set is unreleased (see `notes/findings.md`).
- **Synthetic accuracy is optimistic**: the same model family generates and judges, and the judge filter keeps only 75% of rows,
  biasing the unsafe class toward blatant requests.
- **Calibration is poor.** Scores saturate, hence the 0.017 threshold and the 19.8% false-positive rate on BeaverTails.
- **Input guard only.** No response moderation, binary labels. Adversarial prompts are handled poorly (49.6% recall on adversarial Spanish PGP).
- **Licence:** the stage-1 source is CC BY-NC 4.0; check before any commercial use.
