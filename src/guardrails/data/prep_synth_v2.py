"""Clean synth_v2: keep rows where BOTH judges (in-config qwen verdict, Jev P(block)) agree with the intended label.
Drops writer refusals, CJK drift, short texts and (normalised) duplicates; then splits 70/15/15 (seed 42).
usage: prep_synth_v2.py <scored.parquet>   (output of judge_v2.py on synth_v2.parquet) -> data/processed/synth_v2.parquet
"""
import json
import sys

import numpy as np
import pandas as pd

SEED = 42
REFUSAL = r"(?i)^\s*(n[aã]o (posso|consigo)|no puedo|lo siento|i can(no|')t|i'm sorry|sorry,|desculpe)"
d = pd.read_parquet(sys.argv[1])
n0 = len(d)
d["text"] = d.text.str.strip().str.strip('"')
v = d.verdict.str.strip().str.lower().str.extract(r"(unsafe|safe)")[0]
jev = np.where(d.jev_p >= 0.5, "unsafe", "safe")
ok = (d.text.str.len() > 8) & (v == d.label) & (jev == d.label)
ok &= ~d.text.str.contains(REFUSAL) & ~d.text.str.contains(r"[぀-ヿ一-鿿]")
stats = {"generated": n0, "both_judges_agree": int(((v == d.label) & (jev == d.label)).sum())}
d = d[ok].copy()
d["key"] = d.text.str.lower().str.replace(r"\W+", "", regex=True)
d = d.drop_duplicates("key")
d["unsafe"] = (d.label == "unsafe").astype(int)
d["lang"] = d.language.map({"English": "en", "Brazilian Portuguese": "pt", "Spanish": "es"})
r = np.random.default_rng(SEED).random(len(d))
d["split"] = np.where(r < 0.70, "train", np.where(r < 0.85, "val", "test"))
d[["text", "split", "unsafe", "lang", "topic", "interface"]].to_parquet("data/processed/synth_v2.parquet")
stats |= {"kept": len(d), "unsafe_rate": float(d.unsafe.mean()), "by_lang": d.lang.value_counts().to_dict(),
          "by_split": d.split.value_counts().to_dict(), "email_share": float(d.text.str.contains(r"(?i)e-?mail|correo").mean())}
json.dump(stats, open("data/processed/synth_v2_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
