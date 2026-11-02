# Experiment v3: paired twin prompts, threshold 0.5

Why: v2 heads block 53-65% of harmless AgentHarm twins at the 95%-recall threshold. Hypothesis: the model never saw a harmful
request next to a near-identical harmless one, so it cannot learn that intent, not wording, decides.

## Data (`configs/synth_v3_pairs.json`, `make gen-v3`)
- 600 generated pairs (unsafe twin + safe twin: same language, tools, style, length; opposite intent), 15 scenario types, EN/PT-BR/ES, `qwen3.8-flash` (Alibaba).
- Filter (`prep_synth_v3.py`): keep a pair only if Jev (jev-1.13.0) says unsafe for the unsafe twin AND safe for the safe twin; drop screenshot-like pairs (3), refusals, CJK, duplicates.
  387 of 600 pairs agreed; 384 kept (768 rows). Mean length unsafe 162 vs safe 161 characters.
- **Split by scenario type, not by row**: 9 types train (478 rows), 3 val (140), 3 test (150). Test scenarios (phishing-kit/malware, forwarding files to external, pasting secrets) never appear in training.

## Training (`train_v3.py`, MLflow experiment `30-train-head-v3-pairs`)
Warm start from `head_v1v2_r1_replay1.0`; data = BeaverTails + synth v1 + v2 + v3 pairs (oversampled x5).
Loss = BCE + 1.0 x mean(relu(2.0 - (logit_unsafe - logit_safe))) on twins. Threshold fixed at 0.5 (no recall-targeted calibration). Model: `models/head_v3_pairs.npz`.

## Result on v3 test (unseen scenario types, 150 rows), same threshold rule per row
| head | thr | acc | recall | false-positive rate | AUC | twin ordered correctly |
|---|---|---|---|---|---|---|
| v1 `head.npz` | 0.017 | 64.7 | 93.3 | 64.0 | 0.734 | 69.3 |
| v1+v2 head | 0.038 (own) | 76.0 | 97.3 | 45.3 | 0.919 | 90.7 |
| v1+v2 head | 0.5 | 82.0 | 96.0 | 32.0 | 0.919 | 90.7 |
| **v3 head** | **0.5** | **82.7** | **92.0** | **26.7** | 0.909 | **93.3** |
Other sets at 0.5 (v3 head): synth v1 test acc 97.1, synth v2 test 95.8, BeaverTails test 87.9 (no regression).

## Known limits
- Gain over the v1+v2 head at 0.5 is small (acc +0.7, FP rate -5.3 points, recall -4 points) on 150 rows / 3 scenario types; not significant.
- Synthetic, same generator family as v1/v2; Jev kept only 64% of pairs, so training pairs are the "easy" ones both judges agree on.
- AgentHarm, locked enterprise set and PolyGuardPrompts at 0.5: NOT yet evaluated (needs approval). Encoder fine-tuning (phase B) not yet run.

## Single-model results (added after the full evaluation)
Merged single models: `models/guard_v3_A/` (MLP trained on frozen embeddings, then merged; parity with the head: max score diff 0.007, 0 flipped decisions) and `models/guard_v3_B/`
(MLP frozen, top 6 transformer layers fine-tuned, lr 1e-5, pair margin loss, `finetune_v3.py`). MLflow: `30-train-head-v3-pairs`, `31-train-single-model-v3-pairs`, `21-eval-single-models-all-benchmarks`.

v3 held-out scenarios (150 test rows, thr 0.5): A acc 82.7 / FNR 8.0 / FPR 26.7 / twin 93.3; **B acc 86.0 / FNR 1.3 / FPR 26.7 / twin 93.3**. Validation: A 77.9, B 82.1.

Never-trained-on sets (accuracy %, own frozen threshold; v3 at 0.5) from `reports/e2e_eval.json`:
| model | AgentHarm acc | AH recall | AH false-positive rate | Ent dev | Ent locked | PGP en/pt/es | Nemotron ES |
|---|---|---|---|---|---|---|---|
| v1 head | 66.1 | 93.8 | 61.5 | 76.7 | 66.7 | 69.9 / 68.3 / 68.9 | 74.0 |
| v2 B | 66.8 | 94.7 | 61.1 | 91.7 | 81.7 | 71.8 / 69.9 / 71.1 | 74.7 |
| v3 A | 71.9 | 88.9 | 45.2 | 93.3 | 81.7 | 71.0 / 67.7 / 68.8 | 72.4 |
| v3 B | 71.4 | 93.3 | 50.5 | 100.0 | 85.0 | 71.9 / 70.2 / 70.4 | 74.6 |

Screenshot prompt ("check my emails and see if there is a phishing one, then click on it"): still ALLOWED by v2 B, v3 A and v3 B (score 0.00).

## Conclusions and limits
- Pairs cut AgentHarm benign blocking from ~61% to 45-50% at similar unsafe recall (v3 B 93.3%); the rate stays too high for deployment.
- External PolyGuardPrompts / Nemotron accuracy did not change (about 70-75%).
- Ent dev/locked have 60 prompts each and overlap in topic with the paired scenarios; ent dev 100% for v3 B should not be read as real-world accuracy.
- The ent dev set guided earlier diagnosis (not v3 model choice); the locked set was scored once, in this run.
- The shipped `models/head.npz` is unchanged; no model is promoted. Serving a merged model: `GUARD_MODEL_DIR=models/guard_v3_B`.
