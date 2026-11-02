"""External evaluation / training sets, prompts only (input guard).

- PolyGuardPrompts (CC BY 4.0, test split, machine-translated): EN / PT / ES rows -> data/processed/pgp_{en,pt,es}.parquet, all split=test
  (never used for training or thresholding).
- Nemotron-Safety-Guard-Dataset-v3 Spanish (CC BY 4.0): train capped at NEM_TRAIN_CAP (seeded), valid, test -> data/processed/nem_es.parquet
Columns: text, split, unsafe, lang, adv (1 = adversarial / jailbreak-style prompt when the source says so).
"""
import json

import pandas as pd

SEED, NEM_TRAIN_CAP = 42, 8000
pg = pd.read_parquet("data/raw/polyguard/data/test-00000-of-00001.parquet")
stats = {}
for name, lang, code in (("English", "en", "en"), ("Portuguese", "pt", "pt"), ("Spanish", "es", "es")):
    d = pg[(pg.language == name) & pg.prompt.str.strip().ne("")]
    o = pd.DataFrame({"text": d.prompt.str.strip(), "split": "test", "unsafe": (d.prompt_label == "unsafe").astype(int),
                      "lang": lang, "adv": d.adversarial.astype(int)})
    o.to_parquet(f"data/processed/pgp_{code}.parquet")
    stats[f"pgp_{code}"] = {"n": len(o), "unsafe": int(o.unsafe.sum()), "adv": int(o.adv.sum())}

parts = []
for s, f in (("train", "train"), ("val", "valid"), ("test", "test")):
    d = pd.read_json(f"data/raw/nemotron_v3/es/{f}.jsonl", lines=True)
    d = d[d.prompt.fillna("").str.strip().ne("")].drop_duplicates("prompt")
    if s == "train" and len(d) > NEM_TRAIN_CAP:
        d = d.sample(NEM_TRAIN_CAP, random_state=SEED)
    parts.append(pd.DataFrame({"text": d.prompt.str.strip(), "split": s, "unsafe": (d.prompt_label == "unsafe").astype(int),
                               "lang": "es", "adv": (d.tag == "jailbreaking").astype(int)}))
nem = pd.concat(parts, ignore_index=True)
# a prompt must not sit in two splits: keep the copy in the most held-out split (test > val > train)
rank = {"test": 0, "val": 1, "train": 2}
nem = nem.assign(r=nem.split.map(rank)).sort_values("r").drop_duplicates("text").drop(columns="r").reset_index(drop=True)
assert not nem.groupby("text").split.nunique().gt(1).any()
nem.to_parquet("data/processed/nem_es.parquet")
stats["nem_es"] = {s: {"n": int((nem.split == s).sum()), "unsafe": int(nem[nem.split == s].unsafe.sum())} for s in ("train", "val", "test")}
stats["nem_tags_test"] = pd.read_json("data/raw/nemotron_v3/es/test.jsonl", lines=True).tag.value_counts().to_dict()
json.dump(stats, open("data/processed/ext_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
