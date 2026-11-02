"""Phase B: freeze the MLP, fine-tune only the top K transformer layers of bge-m3 (small LR) on text.
Data = v2 train + a BeaverTails replay sample; threshold re-picked on pooled val (target recall), never on test.
usage: finetune_e2e.py <init_head.npz> <out_dir> [K=6] [lr=1e-5] [epochs=2] [replay_x=1.0] [syn=synth,synth_v2]  (comma list of data/processed files)
Memory (6 GB GPU): fp32 master weights + fp16 autocast; embeddings and lower layers frozen (no grads, no optimiser state).
"""
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from guardrails.core.guard_net import GuardNet, tokenizer
from guardrails.tracking import start

head, out = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
REPLAY_X = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
SYN = sys.argv[7] if len(sys.argv) > 7 else "synth,synth_v2"
BS, ACC, MAXLEN, TARGET, SEED = 8, 2, 192, 0.95, 0
P = "data/processed/"
dev = "cuda"
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

bt = pd.concat([pd.read_parquet(P + f"stage1_{l}.parquet") for l in ("en", "pt")])
sy = pd.concat([pd.read_parquet(P + f"{n}.parquet") for n in SYN.split(",")])  # comma list, e.g. synth,synth_v2
sp = lambda d, s: d[d.split == s][["text", "unsafe"]]
tr = pd.concat([sp(sy, "train"), sp(bt, "train").sample(int(REPLAY_X * len(sp(sy, "train"))), random_state=SEED)]).reset_index(drop=True)
val = pd.concat([sp(sy, "val"), sp(bt, "val")]).reset_index(drop=True)

net = GuardNet.from_head(head)
net.max_len = MAXLEN
for p in net.parameters():
    p.requires_grad = False
top = net.enc.encoder.layer[-K:]
for p in top.parameters():
    p.requires_grad = True
net.to(dev).eval()
top.train()  # dropout only where weights move
tok = tokenizer()
params = [p for p in net.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
steps = EPOCHS * (len(tr) // (BS * ACC))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
scaler = torch.amp.GradScaler()


def pick_thr(y, s, target=TARGET):
    ok = [t for t in np.unique(s) if (s[y == 1] >= t).mean() >= target]
    return float(max(ok)) if ok else 0.5


with start("20-train-single-model-v2", out.rstrip("/").split("/")[-1]):
    mlflow.log_params({"init_head": head, "top_k_layers": K, "lr": LR, "epochs": EPOCHS, "replay_x": REPLAY_X, "syn": SYN,
                       "n_train": len(tr), "trainable_params": sum(p.numel() for p in params), "mlp_frozen": True})
    step = 0
    for ep in range(EPOCHS):
        idx = rng.permutation(len(tr))
        for i in range(0, len(idx) - BS + 1, BS):
            b = tr.iloc[idx[i:i + BS]]
            enc = tok(b.text.tolist(), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = net(enc["input_ids"], enc["attention_mask"])
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.tensor(b.unsafe.values, dtype=torch.float32, device=dev)) / ACC
            scaler.scale(loss).backward()
            if (i // BS + 1) % ACC == 0:
                scaler.unscale_(opt), torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt), scaler.update(), opt.zero_grad(set_to_none=True), sched.step()
                step += 1
                if step % 50 == 0:
                    mlflow.log_metric("loss", float(loss) * ACC, step=step)
                    print(f"ep{ep} step {step}/{steps} loss {float(loss) * ACC:.4f}", flush=True)
    top.eval()
    pv = net.prob(val.text.tolist(), tok, device=dev)
    yv = val.unsafe.values
    net.thr = pick_thr(yv, pv)
    mlflow.log_metrics({"threshold": net.thr, "val_pooled_acc": float(((pv >= net.thr) == yv).mean())})
    net.save(out)
    print("saved", out, "thr", net.thr, "val_pooled_acc", float(((pv >= net.thr) == yv).mean()))
