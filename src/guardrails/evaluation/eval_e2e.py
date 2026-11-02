"""Evaluate single models (GuardNet dirs, or a head .npz merged on the fly) on every held-out test set, from raw text, at each
model's own frozen threshold. Same sets as eval_heads.py plus hand-written enterprise prompts. Logs to MLflow `e2e_eval`.
usage: eval_e2e.py <model_dir|head.npz> [...]     -> reports/e2e_eval.json
"""
import gc
import json
import os
import sys

import mlflow
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from guardrails.core.guard_net import GuardNet, tokenizer
from guardrails.tracking import start

P = "data/processed/"


def sets():
    def te(f, **kw):
        d = pd.read_parquet(P + f)
        return d[d.split == "test"]
    out = {}
    bt = pd.concat([te("stage1_en.parquet"), te("stage1_pt.parquet")])
    out["bt_test"] = bt
    out["syn_test"] = te("synth.parquet")
    for l in ("en", "pt", "es"):
        out[f"syn_{l}"] = out["syn_test"][out["syn_test"].lang == l]
    if os.path.exists(P + "synth_v2.parquet"):
        out["syn_v2"] = te("synth_v2.parquet")
    for n in ("ent_hand", "ent_dev", "ent_locked"):  # ent_hand = 60 prompts already looked at (report-only); ent_locked: do not read errors
        if os.path.exists(P + f"{n}.parquet") and (n != "ent_locked" or os.environ.get("EVAL_LOCKED") == "1"):  # locked set only on request
            out[n] = te(f"{n}.parquet")
    if os.path.exists(P + "agentharm.parquet"):  # public agentic benchmark (eval only, never trained on): harmful vs matched benign tasks
        ah = te("agentharm.parquet")
        out["agentharm"], out["agentharm_nohint"] = ah, ah[ah.hint == 0]
    for l in ("en", "pt", "es"):
        d = te(f"pgp_{l}.parquet")
        out[f"pgp_{l}"], out[f"pgp_{l}_adv"] = d, d[d.adv == 1]
    d = te("nem_es.parquet")
    out["nem_es"], out["nem_es_jailbreak"] = d, d[d.adv == 1]
    return out


def evaluate(net, S, tok, device):
    r = {"threshold": net.thr}
    for name, d in S.items():
        p = net.prob(d.text.tolist(), tok, device=device)
        y, pred = d.unsafe.values.astype(int), p >= net.thr
        r |= {f"{name}_n": len(y), f"{name}_acc": float(accuracy_score(y, pred)), f"{name}_recall": float(pred[y == 1].mean()),
              f"{name}_fpr": float(pred[y == 0].mean()), f"{name}_f1": float(f1_score(y, pred)), f"{name}_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")}
    return r


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S, tok, res = sets(), tokenizer(), {}
    for m in sys.argv[1:]:
        net = GuardNet.from_head(m) if m.endswith(".npz") else GuardNet.load(m)
        res[os.path.basename(m.rstrip("/"))] = r = evaluate(net, S, tok, dev)
        with start("21-eval-single-models-all-benchmarks", os.path.basename(m.rstrip("/")).removesuffix(".npz")):
            mlflow.log_param("model", m)
            mlflow.log_metrics({k: v for k, v in r.items() if not k.endswith("_n") and v == v})
        del net
        gc.collect(), torch.cuda.empty_cache()
    json.dump(res, open("reports/e2e_eval.json", "w"), indent=1)
    keys = [k for k in ("bt_test", "syn_test", "syn_v2", "ent_hand", "ent_dev", "ent_locked", "agentharm", "agentharm_nohint", "pgp_en", "pgp_pt", "pgp_es", "pgp_es_adv", "nem_es", "nem_es_jailbreak") if f"{k}_acc" in next(iter(res.values()))]
    print("model".ljust(28), *[k.ljust(10) for k in keys])
    for h, r in res.items():
        print(h.ljust(28), *[f"{100 * r[k + '_acc']:.1f}".ljust(10) for k in keys])
