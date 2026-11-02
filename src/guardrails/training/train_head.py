"""Train a head on cached embeddings; every run is logged to MLflow. Threshold is picked on val, never on test.

usage: train_head.py <run_name> <head: logreg|mlp> <npz> [<npz> ...]   (train+val+test splits come from the files)
"""
import sys

import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, fbeta_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier

from guardrails.tracking import start


def load(paths):
    z = [np.load(p) for p in paths]
    return {k: np.concatenate([a[k] for a in z]) for k in ("x", "y", "split", "lang")}


def recall_at_fpr(y, s, fpr_t):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_t].max())


def report(prefix, y, s, thr):
    p = (s >= thr).astype(int)
    return {f"{prefix}_acc": accuracy_score(y, p), f"{prefix}_auc": roc_auc_score(y, s),
            f"{prefix}_prauc": average_precision_score(y, s), f"{prefix}_f2": fbeta_score(y, p, beta=2),
            f"{prefix}_recall": float((p[y == 1] == 1).mean()), f"{prefix}_fpr": float((p[y == 0] == 1).mean()),
            f"{prefix}_recall_at_fpr5": recall_at_fpr(y, s, 0.05)}


def main(run_name, head, paths, target_recall=0.95):
    d = load(paths)
    x = d["x"].astype("float32")
    tr, va, te = (d["split"] == s for s in ("train", "val", "test"))
    clf = (LogisticRegression(C=4.0, max_iter=2000, class_weight={0: 1, 1: 1.5}) if head == "logreg"
           else MLPClassifier((512, 128), early_stopping=True, random_state=0, max_iter=200))
    with start("01-train-head-beavertails", run_name):
        mlflow.log_params({"head": head, "files": ",".join(paths), "n_train": int(tr.sum()), "target_recall": target_recall})
        clf.fit(x[tr], d["y"][tr])
        s = clf.predict_proba(x)[:, 1]
        # lowest threshold on val that still reaches the target recall (minimise false negatives), else 0.5
        yv, sv = d["y"][va], s[va]
        ok = [t for t in np.unique(sv) if (sv[yv == 1] >= t).mean() >= target_recall]
        thr = float(max(ok)) if ok else 0.5
        mlflow.log_metric("threshold", thr)
        m = report("val", yv, sv, thr)
        m.update(report("test", d["y"][te], s[te], thr))
        for lang in np.unique(d["lang"]):
            mk = te & (d["lang"] == lang)
            m.update({f"{k}_{lang}": v for k, v in report("test", d["y"][mk], s[mk], thr).items()})
        mlflow.log_metrics(m)
        print({k: round(v, 4) for k, v in m.items() if k.startswith(("val_", "test_acc", "test_auc", "test_recall"))})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
