"""Stage 3: stage-1 head -> fine-tune on a mix of sources (synthetic and/or the Nemotron-v3 Spanish train split) + stage-1 replay.

usage: stage3.py <tag> <replay_fraction> <sources: comma list of syn,nem> [target_recall=0.95]
Threshold = highest value reaching the target unsafe recall (default 95%) on the pooled validation sets of the chosen sources (+ BeaverTails val).
PolyGuardPrompts is never trained on or used for thresholding; evaluate it with eval_heads.py.
"""
import os
import sys

import mlflow
import numpy as np
from sklearn.neural_network import MLPClassifier

from guardrails.tracking import start
from guardrails.training.train_head import load

tag, replay, sources = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(",")
TARGET = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
FILES = {"bt": ["embeddings/stage1_en_bgem3.npz", "embeddings/stage1_pt_bgem3.npz"], "syn": ["embeddings/synth_bgem3.npz"],
         "nem": ["embeddings/nem_es_bgem3.npz"]}
D = {k: load(v) for k, v in FILES.items() if k == "bt" or k in sources}
f32 = lambda a: a.astype("float32")
part = lambda k, s: (f32(D[k]["x"][D[k]["split"] == s]), D[k]["y"][D[k]["split"] == s])
rng = np.random.default_rng(0)


def pick_thr(y, s, target=0.95):
    ok = [t for t in np.unique(s) if (s[y == 1] >= t).mean() >= target]
    return float(max(ok)) if ok else 0.5


with start("03-train-head-spanish", f"{tag}_replay{replay}"):
    mlflow.log_params({"sources": ",".join(sources), "replay": replay, "target_recall": TARGET})
    clf = MLPClassifier((512, 128), early_stopping=True, random_state=0, max_iter=200, warm_start=True)
    xb, yb = part("bt", "train")
    clf.fit(xb, yb)  # stage 1 (identical to stage2.py)
    k = int(replay * len(xb))
    pick = rng.choice(len(xb), k, replace=False) if k else np.array([], dtype=int)
    xs = [xb[pick]] + [part(s, "train")[0] for s in sources]
    ys = [yb[pick]] + [part(s, "train")[1] for s in sources]
    clf.set_params(max_iter=60, early_stopping=False, learning_rate_init=3e-4)
    clf.best_loss_, clf._no_improvement_count = np.inf, 0  # see stage2.py
    clf.fit(np.vstack(xs), np.concatenate(ys))
    mlflow.log_params({f"n_train_{s}": len(part(s, "train")[1]) for s in sources} | {"n_replay": k})
    vx = [part(s, "val") for s in ["bt"] + sources]
    yv = np.concatenate([v[1] for v in vx])
    pv = clf.predict_proba(np.vstack([v[0] for v in vx]))[:, 1]
    thr = pick_thr(yv, pv, TARGET)
    mlflow.log_metric("threshold", thr)
    mlflow.log_metric("val_pooled_acc", float(((pv >= thr) == yv).mean()))
    os.makedirs("models", exist_ok=True)
    np.savez(f"models/head_{tag}.npz", thr=np.float32(thr), **{f"w{i}": w for i, w in enumerate(clf.coefs_)},
             **{f"b{i}": b for i, b in enumerate(clf.intercepts_)})
    print(tag, "thr", thr, "val_pooled_acc", float(((pv >= thr) == yv).mean()))
