"""Evaluate saved heads on every held-out test set at each head's own frozen threshold (no tuning); log to MLflow `external_eval`.

usage: eval_heads.py <head.npz> [<head.npz> ...]
Sets: BeaverTails test (EN+PT), synthetic test (EN/PT/ES), PolyGuardPrompts EN/PT/ES (all + adversarial / non-adversarial),
Nemotron-v3 ES test (all + jailbreaking / non-jailbreaking).
"""
import json
import os
import sys

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from guardrails.tracking import start


def forward(z, x):
    n = sum(k.startswith("w") for k in z.files)
    h = x.astype("float32")
    for i in range(n):
        h = h @ z[f"w{i}"] + z[f"b{i}"]
        if i < n - 1:
            h = np.maximum(h, 0)
    return 1 / (1 + np.exp(-h[:, 0]))


def sets():
    """name -> (x, y, mask). Subsets use `adv` when the file has it."""
    def get(path):
        z = np.load(path)
        adv = z["adv"].astype(bool) if "adv" in z.files else np.zeros(len(z["y"]), bool)
        t = z["split"] == "test"
        return z["x"][t], z["y"][t].astype(int), adv[t], z["lang"][t]
    out = {}
    bx = [get(f"embeddings/stage1_{l}_bgem3.npz") for l in ("en", "pt")]
    out["bt_test"] = (np.vstack([b[0] for b in bx]), np.concatenate([b[1] for b in bx]), None)
    x, y, _, lang = get("embeddings/synth_bgem3.npz")
    out["syn_test"] = (x, y, None)
    for l in ("en", "pt", "es"):
        out[f"syn_{l}"] = (x[lang == l], y[lang == l], None)
    for l in ("en", "pt", "es"):
        x, y, adv, _ = get(f"embeddings/pgp_{l}_bgem3.npz")
        out[f"pgp_{l}"] = (x, y, None)
        out[f"pgp_{l}_adv"] = (x[adv], y[adv], None)
        out[f"pgp_{l}_nonadv"] = (x[~adv], y[~adv], None)
    for name, f in (("syn_v2", "synth_v2"), ("ent_hand", "ent_hand")):  # optional: only if embedded (v2 work)
        if os.path.exists(f"embeddings/{f}_bgem3.npz"):
            x, y, _, _ = get(f"embeddings/{f}_bgem3.npz")
            out[name] = (x, y, None)
    x, y, adv, _ = get("embeddings/nem_es_bgem3.npz")
    out["nem_es"] = (x, y, None)
    out["nem_es_jailbreak"] = (x[adv], y[adv], None)
    out["nem_es_nonjailbreak"] = (x[~adv], y[~adv], None)
    return out


if __name__ == "__main__":
    S = sets()
    res = {}
    for head in sys.argv[1:]:
        z = np.load(head)
        thr = float(z["thr"])
        r = {"threshold": thr}
        for name, (x, y, _) in S.items():
            p = forward(z, x)
            pred = p >= thr
            r[f"{name}_n"] = len(y)
            r[f"{name}_acc"] = float(accuracy_score(y, pred))
            r[f"{name}_recall"] = float(pred[y == 1].mean())
            r[f"{name}_fpr"] = float(pred[y == 0].mean())
            r[f"{name}_auc"] = float(roc_auc_score(y, p))
        res[os.path.basename(head)] = r
        with start("10-eval-heads-all-benchmarks", os.path.basename(head).removesuffix(".npz")):
            mlflow.log_param("head", head)
            mlflow.log_metrics({k: v for k, v in r.items() if not k.endswith("_n")})
    json.dump(res, open("reports/external_eval.json", "w"), indent=1)
    keys = [k for k in ("bt_test", "syn_test", "syn_v2", "ent_hand")if f"{k}_acc" in next(iter(res.values()))] + [ "pgp_en", "pgp_pt", "pgp_es", "pgp_es_adv", "nem_es", "nem_es_jailbreak"]
    print("head".ljust(34), *[k.ljust(10) for k in keys])
    for h, r in res.items():
        print(h.ljust(34), *[f"{100 * r[k + '_acc']:.1f}".ljust(10) for k in keys])
