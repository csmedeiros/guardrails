"""Paired set v3: split pairs into rows, then (after Jev) filter and split by SCENARIO so test/val scenarios are never seen in training.

usage: prep_synth_v3.py rows   -> data/processed/v3_rows.parquet            (two rows per pair, for judge_v2.py)
       prep_synth_v3.py final <scored.parquet> -> data/processed/synth_v3.parquet (+ synth_v3_stats.json)
Keep a pair only if Jev says unsafe for the unsafe twin AND safe for the safe twin. Twins always share a split.
Split: whole pair_topics (scenario types) go to train / val / test (9 / 3 / 3 of 15, seed 42), so test measures unseen scenarios.
Pairs that look like the screenshot case (check mail + click) are dropped so that case stays unseen.
"""
import json, re, sys
import numpy as np
import pandas as pd

SEED = 42
SHOT = re.compile(r"(?is)(check|verif|revis|abr|open).{0,60}(e-?mails?|correos?).{0,80}(click|clic|clique)")
REFUSAL = r"(?i)^\s*(n[aã]o (posso|consigo)|no puedo|lo siento|i can(no|')t|i'm sorry|sorry,|desculpe)"
if sys.argv[1] == "rows":
    d = pd.read_parquet("data/synthetic/synth_v3.parquet")
    rows = []
    for i, r in d.iterrows():
        m = re.search(r"UNSAFE:\s*(.*?)\s*\n+\s*SAFE:\s*(.*)", r.pair, re.S)
        if not m: continue
        for lab, t in (("unsafe", m[1]), ("safe", m[2])):
            rows.append(dict(pair_id=i, label=lab, text=t.strip().strip('"'), pair_topic=r.pair_topic, language=r.language, interface=r.interface))
    o = pd.DataFrame(rows); o.to_parquet("data/processed/v3_rows.parquet")
    print("pairs", len(d), "parsed pairs", len(o) // 2, "rows", len(o)); sys.exit()
d = pd.read_parquet(sys.argv[2]); n0 = len(d)
jev = np.where(d.jev_p >= 0.5, "unsafe", "safe")
d["agree"] = jev == d.label
g = d.groupby("pair_id")
keep = g.agree.all() & (g.size() == 2) & ~g.text.apply(lambda s: s.str.contains(SHOT).any()) \
       & g.text.apply(lambda s: (s.str.len() > 8).all() and not s.str.contains(REFUSAL).any() and not s.str.contains(r"[぀-ヿ一-鿿]").any())
stats = {"rows_scored": n0, "pairs_scored": int(g.ngroups), "pairs_both_judges_agree": int(g.agree.all().sum()),
         "pairs_dropped_screenshot_like": int(g.text.apply(lambda s: s.str.contains(SHOT).any()).sum())}
d = d[d.pair_id.isin(keep[keep].index)].copy()
d["key"] = d.text.str.lower().str.replace(r"\W+", "", regex=True)
d = d.drop_duplicates("key")
d = d[d.groupby("pair_id").pair_id.transform("size") == 2]          # a twin lost to dedup -> drop the pair
topics = sorted(d.pair_topic.unique()); rng = np.random.default_rng(SEED); rng.shuffle(topics)
sp = {t: ("train" if i < 9 else "val" if i < 12 else "test") for i, t in enumerate(topics)}
d["split"] = d.pair_topic.map(sp); d["unsafe"] = (d.label == "unsafe").astype(int)
d["lang"] = d.language.map({"English": "en", "Brazilian Portuguese": "pt", "Spanish": "es"})
d["topic"] = d.pair_topic.str.split(" \\| ").str[0]
d[["text", "split", "unsafe", "lang", "topic", "interface", "pair_id"]].to_parquet("data/processed/synth_v3.parquet")
stats |= {"kept_pairs": int(d.pair_id.nunique()), "kept_rows": len(d), "by_split_rows": d.split.value_counts().to_dict(),
          "by_lang": d.lang.value_counts().to_dict(), "heldout_val_topics": [t for t in topics if sp[t] == "val"],
          "heldout_test_topics": [t for t in topics if sp[t] == "test"],
          "mean_len_unsafe": float(d[d.unsafe == 1].text.str.len().mean()), "mean_len_safe": float(d[d.unsafe == 0].text.str.len().mean())}
json.dump(stats, open("data/processed/synth_v3_stats.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, indent=1, ensure_ascii=False))
