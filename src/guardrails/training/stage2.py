"""Stage 1 (pretrain head on BeaverTails EN/PT) -> stage 2 (fine-tune same head on synthetic EN/PT/ES + a replay share).

Threshold is picked on the pooled validation sets (target recall) and applied unchanged to the never-seen test sets.
usage: stage2.py <replay_fraction> <run_tag>
"""
import os
import sys

import mlflow
import numpy as np
from sklearn.neural_network import MLPClassifier

from guardrails.tracking import start
from guardrails.training.train_head import load, report

replay, tag = float(sys.argv[1]), sys.argv[2]
bt = load(["embeddings/stage1_en_bgem3.npz", "embeddings/stage1_pt_bgem3.npz"])
sy = load(os.environ.get("SYN_EMB", "embeddings/synth_bgem3.npz").split(","))  # comma list; v2: SYN_EMB=embeddings/synth_v2_bgem3.npz
f32 = lambda a: a.astype("float32")
rng = np.random.default_rng(0)


def part(d, s):
    m = d["split"] == s
    return f32(d["x"][m]), d["y"][m], d["lang"][m]


def pick_thr(y, s, target=0.95):
    ok = [t for t in np.unique(s) if (s[y == 1] >= t).mean() >= target]
    return float(max(ok)) if ok else 0.5


with start("02-train-head-synthetic-v1", f"{tag}_replay{replay}"):
    clf = MLPClassifier((512, 128), early_stopping=True, random_state=0, max_iter=200, warm_start=True)
    xb, yb, _ = part(bt, "train")
    clf.fit(xb, yb)  # stage 1
    # stage-1-only numbers on the synthetic test set (how a BeaverTails-only head transfers)
    xst, yst, lst = part(sy, "test")
    mlflow.log_metrics({f"s1only_{k}": v for k, v in report("syn_test", yst, clf.predict_proba(xst)[:, 1], 0.5).items()})
    # stage 2: synthetic train + replay of stage-1 train; fewer, gentler iterations on the same weights
    xs, ys, _ = part(sy, "train")
    k = int(replay * len(xb))
    pick = rng.choice(len(xb), k, replace=False) if k else np.array([], dtype=int)
    clf.set_params(max_iter=60, early_stopping=False, learning_rate_init=3e-4)
    clf.best_loss_, clf._no_improvement_count = np.inf, 0  # early-stopping fit leaves best_loss_ None; reset to continue training
    clf.fit(np.vstack([xs, xb[pick]]), np.concatenate([ys, yb[pick]]))
    mlflow.log_params({"replay": replay, "n_syn_train": len(xs), "n_replay": k, "n_bt_train": len(xb)})
    xsv, ysv, _ = part(sy, "val")
    xbv, ybv, _ = part(bt, "val")
    pv = clf.predict_proba(np.vstack([xsv, xbv]))[:, 1]
    thr = pick_thr(np.concatenate([ysv, ybv]), pv)
    mlflow.log_metric("threshold", thr)
    yv_all = np.concatenate([ysv, ybv])
    mlflow.log_metrics({"val_pooled_acc": float(((pv >= thr) == yv_all).mean()),
                        "val_syn_acc": float(((pv[:len(ysv)] >= thr) == ysv).mean()),
                        "val_bt_acc": float(((pv[len(ysv):] >= thr) == ybv).mean())})
    os.makedirs("models", exist_ok=True)
    wpath = f"models/head_{tag}_replay{replay}.npz"  # plain arrays, no pickle
    np.savez(wpath, thr=np.float32(thr), **{f"w{i}": w for i, w in enumerate(clf.coefs_)},
             **{f"b{i}": b for i, b in enumerate(clf.intercepts_)})
    mlflow.log_artifact(wpath)
    out = {}
    for name, d in (("syn_test", sy), ("bt_test", bt)):
        x, y, lg = part(d, "test")
        s = clf.predict_proba(x)[:, 1]
        out.update(report(name, y, s, thr))
        for lang in np.unique(lg):
            m = lg == lang
            out.update({f"{kk}_{lang}": v for kk, v in report(name, y[m], s[m], thr).items()})
    mlflow.log_metrics(out)
    print({k: round(v, 4) for k, v in out.items() if k.split("_")[-1] in ("acc", "auc", "recall") or k.endswith(("test_acc", "test_auc"))})
