"""Detailed baseline: torch MLP head on frozen bge-m3 embeddings, per-epoch curves + weight distributions logged to MLflow.

usage: baseline_torch.py <run_name> <npz> [<npz> ...]
Early stopping and the operating threshold use val only; test is touched once at the end.
"""
import copy
import sys

import matplotlib
import mlflow
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix,
                             roc_auc_score)
from torch import nn

from guardrails.tracking import start
from guardrails.training.train_head import load, report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EPOCHS, BS, LR, WD, DROP, POS_W, PATIENCE, TARGET_RECALL = 40, 256, 1e-3, 1e-2, 0.1, 1.5, 6, 0.95
torch.manual_seed(0)
np.random.seed(0)


def make_head(d_in):
    return nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(DROP), nn.Linear(512, 128), nn.ReLU(),
                         nn.Dropout(DROP), nn.Linear(128, 1))


def weights(m):
    return {f"layer{i}": l.weight.detach().flatten().numpy().copy() for i, l in enumerate(m) if isinstance(l, nn.Linear)}


def evaluate(m, x, y, lossf):
    m.eval()
    with torch.no_grad():
        z = m(x).squeeze(1)
        loss = lossf(z, y).item()
    return loss, torch.sigmoid(z).numpy()


def main(run_name, paths):
    d = load(paths)
    tr, va, te = (d["split"] == s for s in ("train", "val", "test"))
    T = lambda a: torch.tensor(a.astype("float32"))
    x, y = T(d["x"]), T(d["y"])
    m = make_head(x.shape[1])
    w_init = weights(m)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_W))
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    xt, yt, xv, yv = x[tr], y[tr], x[va], y[va]
    hist, best, best_state, bad = [], 1e9, None, 0

    with start("01-train-head-beavertails", run_name):
        mlflow.log_params({"head": "torch_mlp_1024-512-128-1", "epochs_max": EPOCHS, "batch": BS, "lr": LR,
                           "weight_decay": WD, "dropout": DROP, "pos_weight": POS_W, "patience": PATIENCE,
                           "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                           "extractor": "BAAI/bge-m3 (frozen, fp16, max_len 256)", "files": ",".join(paths)})
        for ep in range(1, EPOCHS + 1):
            m.train()
            perm = torch.randperm(len(xt))
            run = 0.0
            for i in range(0, len(xt), BS):
                idx = perm[i:i + BS]
                opt.zero_grad()
                l = lossf(m(xt[idx]).squeeze(1), yt[idx])
                l.backward()
                opt.step()
                run += l.item() * len(idx)
            tl, ptr = evaluate(m, xt, yt, lossf)
            vl, pv = evaluate(m, xv, yv, lossf)
            row = {"train_loss": run / len(xt), "val_loss": vl,
                   "train_acc": float(((ptr >= .5) == yt.numpy()).mean()), "val_acc": float(((pv >= .5) == yv.numpy()).mean()),
                   "train_recall": float((ptr[yt.numpy() == 1] >= .5).mean()), "val_recall": float((pv[yv.numpy() == 1] >= .5).mean()),
                   "val_auc": float(roc_auc_score(yv.numpy(), pv)),
                   "weight_norm_l0": float(m[0].weight.norm()), "weight_norm_l1": float(m[3].weight.norm()), "weight_norm_l2": float(m[6].weight.norm())}
            hist.append(row)
            mlflow.log_metrics(row, step=ep)
            if vl < best - 1e-4:
                best, best_state, bad = vl, copy.deepcopy(m.state_dict()), 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        m.load_state_dict(best_state)
        best_ep = int(np.argmin([h["val_loss"] for h in hist])) + 1
        mlflow.log_params({"best_epoch": best_ep, "epochs_run": len(hist)})

        # operating threshold from val: highest threshold that still reaches the target recall
        _, pv = evaluate(m, xv, yv, lossf)
        ok = [t for t in np.unique(pv) if (pv[yv.numpy() == 1] >= t).mean() >= TARGET_RECALL]
        thr = float(max(ok)) if ok else 0.5
        mlflow.log_metric("threshold", thr)
        _, ps = evaluate(m, x, y, lossf)
        yy = d["y"]
        out = report("val", yy[va], ps[va], thr)
        out.update(report("test", yy[te], ps[te], thr))
        out.update({k.replace("test_", "test_thr05_"): v for k, v in report("test", yy[te], ps[te], 0.5).items()})
        for lang in np.unique(d["lang"]):
            mk = te & (d["lang"] == lang)
            out.update({f"{k}_{lang}": v for k, v in report("test", yy[mk], ps[mk], thr).items()})
        mlflow.log_metrics(out)

        ep_ = np.arange(1, len(hist) + 1)
        col = lambda k: [h[k] for h in hist]

        def curve(name, keys, ylabel):
            f, a = plt.subplots(figsize=(5.5, 3.6))
            for k, lab in keys:
                a.plot(ep_, col(k), label=lab)
            a.axvline(best_ep, color="gray", ls=":", label=f"best epoch ({best_ep})")
            a.set_xlabel("epoch"); a.set_ylabel(ylabel); a.legend(); a.grid(alpha=.3)
            f.tight_layout(); mlflow.log_figure(f, f"plots/{name}.png"); plt.close(f)

        curve("loss", [("train_loss", "train"), ("val_loss", "val")], "BCE loss")
        curve("accuracy", [("train_acc", "train"), ("val_acc", "val")], "accuracy (thr 0.5)")
        curve("recall", [("train_recall", "train"), ("val_recall", "val")], "unsafe recall (thr 0.5)")
        curve("val_auc", [("val_auc", "val ROC AUC")], "ROC AUC")
        curve("weight_norms", [("weight_norm_l0", "layer0"), ("weight_norm_l1", "layer1"), ("weight_norm_l2", "layer2")], "L2 norm")

        w_fin = weights(m)
        f, ax = plt.subplots(1, len(w_fin), figsize=(4 * len(w_fin), 3.4))
        for a, k in zip(np.atleast_1d(ax), w_fin):
            a.hist(w_init[k], bins=80, alpha=.5, label="init", density=True)
            a.hist(w_fin[k], bins=80, alpha=.5, label="trained", density=True)
            a.set_title(f"{k}  (std {w_init[k].std():.3f} -> {w_fin[k].std():.3f})"); a.set_xlabel("weight"); a.legend()
        f.tight_layout(); mlflow.log_figure(f, "plots/weight_distributions.png"); plt.close(f)
        mlflow.log_metrics({f"w_std_{k}": float(v.std()) for k, v in w_fin.items()})
        mlflow.log_metrics({f"w_absmax_{k}": float(np.abs(v).max()) for k, v in w_fin.items()})

        yte, ste = yy[te], ps[te]
        f, a = plt.subplots(figsize=(4.5, 4)); RocCurveDisplay.from_predictions(yte, ste, ax=a); mlflow.log_figure(f, "plots/roc_test.png"); plt.close(f)
        f, a = plt.subplots(figsize=(4.5, 4)); PrecisionRecallDisplay.from_predictions(yte, ste, ax=a); mlflow.log_figure(f, "plots/pr_test.png"); plt.close(f)
        f, a = plt.subplots(figsize=(4.2, 4)); ConfusionMatrixDisplay(confusion_matrix(yte, ste >= thr), display_labels=["safe", "unsafe"]).plot(ax=a, colorbar=False)
        a.set_title(f"test, thr={thr:.3f}"); mlflow.log_figure(f, "plots/confusion_test.png"); plt.close(f)
        f, a = plt.subplots(figsize=(5, 3.4)); a.hist([ste[yte == 0], ste[yte == 1]], bins=40, label=["safe", "unsafe"], stacked=True)
        a.axvline(thr, color="k", ls="--", label="threshold"); a.set_xlabel("P(unsafe)"); a.legend(); f.tight_layout()
        mlflow.log_figure(f, "plots/score_distribution_test.png"); plt.close(f)
        torch.save(m.state_dict(), "mlflow/baseline_head.pt")
        mlflow.log_artifact("mlflow/baseline_head.pt")
        print({k: round(v, 4) for k, v in out.items() if k in ("val_acc", "test_acc", "test_auc", "test_recall", "test_fpr",
                                                               "test_recall_at_fpr5", "test_thr05_acc")}, "best_epoch", best_ep)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
