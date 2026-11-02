"""Paired enterprise set v3 (unsafe/safe twins) from configs/synth_v3_pairs.json.

usage: gen_synth_v3.py <num_pairs> [preview]   (Alibaba endpoint from .env)
"""
import os
import sys

import data_designer.config as dd
from data_designer.interface import DataDesigner

for line in open(".env"):  # OPENAI_API_KEY / OPENAI_BASE_URL (Alibaba Model Studio), never printed
    k, _, v = line.strip().partition("=")
    if k and not k.startswith("#"):
        os.environ.setdefault(k, v.strip().strip("\"'"))
provider = dd.ModelProvider(name="alibaba", endpoint=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
b = dd.DataDesignerConfigBuilder.from_config("configs/synth_v3_pairs.json")
n, prev = int(sys.argv[1]), "preview" in sys.argv
d = DataDesigner(artifact_path="data/synthetic/dd_artifacts", model_providers=[provider])
res = d.preview(b, num_records=n) if prev else d.create(b, num_records=n, dataset_name="synth_v3")
df = res.dataset if prev else res.load_dataset()
out = "data/synthetic/preview_v3.parquet" if prev else "data/synthetic/synth_v3.parquet"
df.to_parquet(out)
print(df[["language", "pair_topic", "pair"]].to_string(max_colwidth=200))
print("saved", out, len(df))
