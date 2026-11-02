"""Head v3: warm-start from the v1+v2 head, train on BeaverTails + synth v1 + v2 + paired v3 with a PAIR MARGIN loss, threshold fixed at 0.5.

loss = BCE(all rows) + LAM * mean(relu(MARGIN - (logit_unsafe_twin - logit_safe_twin)))   (twins share a split and a scenario)
v3 validation/test are scenario types never seen in training (see prep_synth_v3.py). No recall-targeted threshold: thr = 0.5.
usage: train_v3.py <init_head.npz> <out_head.npz> [lam=1.0] [margin=2.0] [epochs=40] [pair_x=5]
"""
import json, os, sys
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from guardrails.tracking import start
from guardrails.training.train_head import load, report

init, out = sys.argv[1], sys.argv[2]
LAM, MARGIN, EPOCHS, PAIR_X = (float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, float(sys.argv[4]) if len(sys.argv) > 4 else 2.0,
                               int(sys.argv[5]) if len(sys.argv) > 5 else 40, int(sys.argv[6]) if len(sys.argv) > 6 else 5)
torch.manual_seed(0); rng = np.random.default_rng(0)
f32 = lambda a: torch.tensor(a.astype("float32"))
bt = load(["embeddings/stage1_en_bgem3.npz", "embeddings/stage1_pt_bgem3.npz"])
s1, s2 = load(["embeddings/synth_bgem3.npz"]), load(["embeddings/synth_v2_bgem3.npz"])
s3 = load(["embeddings/synth_v3_bgem3.npz"]); pid = pd.read_parquet("data/processed/synth_v3.parquet").pair_id.values
assert len(pid) == len(s3["y"])
sel = lambda d, s: d["split"] == s
z = np.load(init)
net = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.ReLU(), torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
for i, l in enumerate([net[0], net[2], net[4]]):
    l.weight.data = torch.tensor(z[f"w{i}"].T.astype("float32")); l.bias.data = torch.tensor(z[f"b{i}"].astype("float32").ravel())
# general rows: everything except v3 (v3 rows enter as pairs)
gx = np.vstack([d["x"][sel(d, "train")] for d in (bt, s1, s2)]).astype("float32"); gy = np.concatenate([d["y"][sel(d, "train")] for d in (bt, s1, s2)])
m3 = sel(s3, "train"); x3, y3, p3 = s3["x"][m3].astype("float32"), s3["y"][m3], pid[m3]
pairs = [(np.where((p3 == p) & (y3 == 1))[0][0], np.where((p3 == p) & (y3 == 0))[0][0]) for p in np.unique(p3)]
U = torch.tensor([a for a, _ in pairs]); S = torch.tensor([b for _, b in pairs]); X3 = f32(x3)
GX, GY = f32(gx), torch.tensor(gy.astype("float32"))

def ev(d, mask, thr=0.5):
    with torch.no_grad(): s = torch.sigmoid(net(f32(d["x"][mask])).squeeze(1)).numpy()
    return s
def pair_acc(s, y, p):
    ok = [s[(p == q) & (y == 1)][0] > s[(p == q) & (y == 0)][0] for q in np.unique(p)]; return float(np.mean(ok))
def val_stats(split):
    m = sel(s3, split); s = ev(s3, m); y = s3["y"][m]; r = report(f"v3_{split}", y, s, 0.5); r[f"v3_{split}_pair_acc"] = pair_acc(s, y, pid[m]); return r

opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
with start("30-train-head-v3-pairs", os.path.basename(out).removesuffix(".npz")):
    mlflow.log_params({"init": init, "lam": LAM, "margin": MARGIN, "epochs": EPOCHS, "pair_x": PAIR_X, "threshold": 0.5,
                       "n_general_train": len(gy), "n_train_pairs": len(pairs), "split": "by scenario type (9/3/3 of 15)"})
    best, best_state = -1, None
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(GX)); steps = 0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]; loss = F.binary_cross_entropy_with_logits(net(GX[b]).squeeze(1), GY[b])
            for _ in range(int(PAIR_X)):            # pair batch every step (oversampled: the paired set is small)
                q = torch.randint(0, len(pairs), (64,)); lu, ls = net(X3[U[q]]).squeeze(1), net(X3[S[q]]).squeeze(1)
                loss = loss + (F.binary_cross_entropy_with_logits(lu, torch.ones(64)) + F.binary_cross_entropy_with_logits(ls, torch.zeros(64))) / PAIR_X \
                       + LAM * F.relu(MARGIN - (lu - ls)).mean() / PAIR_X
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); v = val_stats("val"); sc = v["v3_val_acc"] + v["v3_val_pair_acc"]
        mlflow.log_metrics({k: x for k, x in v.items() if not k.endswith("auc") or True}, step=ep)
        if sc > best: best, best_state, best_ep = sc, {k: t.clone() for k, t in net.state_dict().items()}, ep
    net.load_state_dict(best_state); net.eval(); mlflow.log_param("best_epoch", best_ep)
    res = {}
    for sp in ("val", "test"): res |= val_stats(sp)
    for name, d in (("syn_v1_test", s1), ("syn_v2_test", s2), ("bt_test", bt)):
        m = sel(d, "test"); res |= report(name, d["y"][m], ev(d, m), 0.5)
    mlflow.log_metrics(res)
    np.savez(out, thr=np.float32(0.5), **{f"w{i}": l.weight.data.numpy().T for i, l in enumerate([net[0], net[2], net[4]])},
             **{f"b{i}": l.bias.data.numpy() for i, l in enumerate([net[0], net[2], net[4]])})
    mlflow.log_artifact(out)
    print(json.dumps({k: round(v, 4) for k, v in res.items() if k.endswith(("acc", "recall", "fpr", "auc"))}, indent=1))
