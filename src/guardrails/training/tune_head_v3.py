"""Optuna search over the MLP head (sizes, depth of the two hidden layers, dropout, lr, weight decay, pair loss) on cached bge-m3 embeddings.

Objective = v3 VALIDATION accuracy + twin-ordering accuracy at threshold 0.5 (val scenario types never seen in training). Test, AgentHarm and the locked
set are NOT used here. Trials run in parallel threads (n_jobs); every trial is logged to MLflow `32-tune-head-v3`.
usage: tune_head_v3.py [n_trials=40] [n_jobs=4]   -> models/head_v3_tuned.npz (best trial by val), reports/tune_head_v3.json
"""
import json, sys, threading
import mlflow, numpy as np, optuna, pandas as pd, torch, torch.nn.functional as F
from guardrails.training.train_head import load
from guardrails.tracking import TRACKING_URI

N, JOBS, EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 40, int(sys.argv[2]) if len(sys.argv) > 2 else 4, 25
dev = "cuda"; sel = lambda d, s: d["split"] == s
T = lambda a: torch.tensor(a.astype("float32"), device=dev)
bt = load(["embeddings/stage1_en_bgem3.npz", "embeddings/stage1_pt_bgem3.npz"]); s1 = load(["embeddings/synth_bgem3.npz"]); s2 = load(["embeddings/synth_v2_bgem3.npz"])
s3 = load(["embeddings/synth_v3_bgem3.npz"]); pid = pd.read_parquet("data/processed/synth_v3.parquet").pair_id.values
GX = T(np.vstack([d["x"][sel(d, "train")] for d in (bt, s1, s2)])); GY = T(np.concatenate([d["y"][sel(d, "train")] for d in (bt, s1, s2)]))
m3 = sel(s3, "train"); y3, p3 = s3["y"][m3], pid[m3]; X3 = T(s3["x"][m3])
ids = np.unique(p3); U = torch.tensor([np.where((p3 == p) & (y3 == 1))[0][0] for p in ids], device=dev); S = torch.tensor([np.where((p3 == p) & (y3 == 0))[0][0] for p in ids], device=dev)
mv = sel(s3, "val"); XV, yv, pv_ = T(s3["x"][mv]), s3["y"][mv], pid[mv]
mt = sel(s3, "test")
best = {"score": -1}; lock = threading.Lock()

def build(h1, h2, drop):
    return torch.nn.Sequential(torch.nn.Linear(1024, h1), torch.nn.ReLU(), torch.nn.Dropout(drop), torch.nn.Linear(h1, h2), torch.nn.ReLU(), torch.nn.Dropout(drop), torch.nn.Linear(h2, 1)).to(dev)

def val_metrics(net):
    net.eval()
    with torch.no_grad(): s = torch.sigmoid(net(XV).squeeze(1)).cpu().numpy()
    pa = float(np.mean([s[(pv_ == q) & (yv == 1)][0] > s[(pv_ == q) & (yv == 0)][0] for q in np.unique(pv_)]))
    return dict(acc=float(((s >= .5) == yv).mean()), pair=pa, fnr=float((s[yv == 1] < .5).mean()), fpr=float((s[yv == 0] >= .5).mean()))

def objective(t):
    h1, h2 = t.suggest_categorical("h1", [256, 512, 768, 1024]), t.suggest_categorical("h2", [64, 128, 256])
    drop, lr = t.suggest_float("dropout", 0.0, 0.3), t.suggest_float("lr", 1e-4, 2e-3, log=True)
    wd, lam = t.suggest_float("wd", 1e-4, 5e-2, log=True), t.suggest_float("lam", 0.0, 2.0)
    margin, px = t.suggest_float("margin", 0.5, 4.0), t.suggest_int("pair_x", 1, 8)
    torch.manual_seed(t.number); net = build(h1, h2, drop); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); bs_best, st_best = -1, None
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(GX), device=dev)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]; loss = F.binary_cross_entropy_with_logits(net(GX[b]).squeeze(1), GY[b])
            for _ in range(px):
                q = torch.randint(0, len(ids), (64,), device=dev); lu, ls = net(X3[U[q]]).squeeze(1), net(X3[S[q]]).squeeze(1)
                loss = loss + (F.binary_cross_entropy_with_logits(lu, torch.ones_like(lu)) + F.binary_cross_entropy_with_logits(ls, torch.zeros_like(ls))) / px \
                       + lam * F.relu(margin - (lu - ls)).mean() / px
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); v = val_metrics(net)
        if v["acc"] + v["pair"] > bs_best: bs_best, st_best, v_best = v["acc"] + v["pair"], {k: x.clone() for k, x in net.state_dict().items()}, v
    for k, x in v_best.items(): t.set_user_attr(k, x)
    with lock:
        mlflow.set_tracking_uri(TRACKING_URI); mlflow.set_experiment("32-tune-head-v3")
        with mlflow.start_run(run_name=f"trial{t.number}"):
            mlflow.log_params(t.params); mlflow.log_metrics({f"val_{k}": x for k, x in v_best.items()})
        if bs_best > best["score"]:
            net.load_state_dict(st_best); best.update(score=bs_best, number=t.number, params=dict(t.params))
            L = [net[0], net[3], net[6]]
            np.savez("models/head_v3_tuned.npz", thr=np.float32(0.5), **{f"w{i}": l.weight.data.cpu().numpy().T for i, l in enumerate(L)}, **{f"b{i}": l.bias.data.cpu().numpy() for i, l in enumerate(L)})
    return bs_best

st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial(dict(h1=512, h2=128, dropout=0.0, lr=3e-4, wd=0.01, lam=1.0, margin=2.0, pair_x=5))   # baseline: same size/settings as the current head, from scratch
optuna.logging.set_verbosity(optuna.logging.WARNING)
st.optimize(objective, n_trials=N, n_jobs=JOBS)
rows = sorted([dict(trial=t.number, score=t.value, **t.params, **t.user_attrs) for t in st.trials], key=lambda r: -r["score"])
json.dump(rows, open("reports/tune_head_v3.json", "w"), indent=1)
print("baseline trial 0:", {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[[r["trial"] for r in rows].index(0)].items()})
for r in rows[:8]: print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
