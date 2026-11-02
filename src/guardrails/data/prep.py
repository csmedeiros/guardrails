"""Stage-1 data prep: agree-labels filter + group split by English prompt (no leakage across splits/languages)."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
df = pd.read_json("data/raw/guardbench-ptbr/guardbench_ptbr.jsonl", lines=True)
n0 = len(df)
agree = df[df.label_en == df.label_pt].copy()
agree["pid"] = agree.prompt_en.str.strip().str.lower().factorize()[0]
stats = {"rows": n0, "agree": len(agree), "dropped_disagree": n0 - len(agree),
         "unique_prompts_agree": int(agree.pid.nunique())}
# one row per prompt: majority label, ties -> unsafe (bias against false negatives)
g = agree.groupby("pid").agg(
    prompt_en=("prompt_en", "first"),
    prompt_pt=("prompt_pt", "first"),
    unsafe=("label_en", lambda s: int((s == "unsafe").mean() >= 0.5)),
)
stats["unsafe_rate"] = float(g.unsafe.mean())
idx = np.arange(len(g))
tv, te = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(idx, groups=idx))
tr, va = next(GroupShuffleSplit(1, test_size=0.15 / 0.85, random_state=SEED).split(tv, groups=tv))
g["split"] = "train"
g.iloc[tv[va], g.columns.get_loc("split")] = "val"
g.iloc[te, g.columns.get_loc("split")] = "test"
for lang in ("en", "pt"):
    out = g[["split", f"prompt_{lang}", "unsafe"]].rename(columns={f"prompt_{lang}": "text"})
    out["lang"] = lang
    out.reset_index().to_parquet(f"data/processed/stage1_{lang}.parquet")
stats["split_sizes"] = g.split.value_counts().to_dict()
json.dump(stats, open("data/processed/stage1_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
