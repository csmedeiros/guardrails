"""One model: bge-m3 encoder (CLS pool + L2 norm) + MLP head -> logit(unsafe). Saved as safetensors + meta.json (no pickle)."""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

BASE = "BAAI/bge-m3"


class GuardNet(nn.Module):
    def __init__(self, base: str = BASE, dims=(1024, 512, 128)):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        d = list(dims)
        layers = []
        for i in range(len(d) - 1):
            layers += [nn.Linear(d[i], d[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers, nn.Linear(d[-1], 1))
        self.thr, self.max_len = 0.5, 256

    def forward(self, input_ids, attention_mask):
        h = F.normalize(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0], dim=-1)
        return self.mlp(h)[:, 0]

    @classmethod
    def from_head(cls, head_npz: str, base: str = BASE):
        z = np.load(head_npz)
        n = sum(k.startswith("w") for k in z.files)
        m = cls(base, dims=[z["w0"].shape[0]] + [z[f"w{i}"].shape[1] for i in range(n - 1)])
        lin = [l for l in m.mlp if isinstance(l, nn.Linear)]
        for i, l in enumerate(lin):  # sklearn coefs are (in, out); nn.Linear wants (out, in)
            l.weight.data = torch.tensor(z[f"w{i}"].T.copy())
            l.bias.data = torch.tensor(z[f"b{i}"].copy())
        m.thr = float(z["thr"])
        return m

    def save(self, out: str):
        os.makedirs(out, exist_ok=True)
        save_file({k: v.contiguous().cpu() for k, v in self.state_dict().items()}, f"{out}/model.safetensors")
        json.dump({"thr": self.thr, "max_len": self.max_len, "base": BASE,
                   "dims": [self.mlp[0].in_features] + [l.out_features for l in self.mlp if isinstance(l, nn.Linear)][:-1]},
                  open(f"{out}/meta.json", "w"))

    @classmethod
    def load(cls, path: str):
        meta = json.load(open(f"{path}/meta.json"))
        m = cls(meta["base"], dims=meta["dims"])
        m.load_state_dict(load_file(f"{path}/model.safetensors"))
        m.thr, m.max_len = meta["thr"], meta["max_len"]
        return m

    @torch.no_grad()
    def prob(self, texts, tok, bs: int = 32, device: str = "cpu") -> np.ndarray:
        self.eval().to(device)
        order = np.argsort([len(t) for t in texts])  # length-sorted batches = less padding
        out = []
        for i in range(0, len(texts), bs):
            b = tok([texts[j] for j in order[i:i + bs]], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                out.append(torch.sigmoid(self(b["input_ids"], b["attention_mask"]).float()).cpu().numpy())
        p = np.empty(len(texts), "float32")
        p[order] = np.concatenate(out)
        return p


def tokenizer():
    return AutoTokenizer.from_pretrained(BASE)
