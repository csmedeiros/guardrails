"""Clean synthetic set: keep rows where the independent judge agrees with the intended label, dedupe, split 70/15/15."""
import json

import numpy as np
import pandas as pd

SEED = 42
d = pd.read_parquet("data/synthetic/synth_v1.parquet")
n0 = len(d)
d["v"] = d.verdict.str.strip().str.lower().str.extract(r"(unsafe|safe)")[0]
d["text"] = d.text.str.strip().str.strip('"')
d = d[(d.text.str.len() > 8) & (d.v == d.label)]
d = d.drop_duplicates("text")
d["unsafe"] = (d.label == "unsafe").astype(int)
d["lang"] = d.language.map({"English": "en", "Brazilian Portuguese": "pt", "Spanish": "es"})
r = np.random.default_rng(SEED).random(len(d))
d["split"] = np.where(r < 0.70, "train", np.where(r < 0.85, "val", "test"))
d[["text", "split", "unsafe", "lang"]].to_parquet("data/processed/synth.parquet")
stats = {"generated": n0, "kept": len(d), "unsafe_rate": float(d.unsafe.mean()),
         "by_lang": d.lang.value_counts().to_dict(), "by_split": d.split.value_counts().to_dict(),
         "kept_by_label": d.label.value_counts().to_dict()}
json.dump(stats, open("data/processed/synth_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
