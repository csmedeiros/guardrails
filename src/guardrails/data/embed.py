"""Embed a parquet of {text, split, unsafe, lang} with a frozen multilingual extractor; cache to embeddings/."""
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-m3"


def embed(parquet: str, out: str, model: str = MODEL, max_len: int = 256, bs: int = 32):
    df = pd.read_parquet(parquet)
    cuda = torch.cuda.is_available()  # CPU works too, just slower
    m = SentenceTransformer(model, device="cuda" if cuda else "cpu",
                            model_kwargs={"torch_dtype": torch.float16} if cuda else {})
    m.max_seq_length = max_len
    x = m.encode(df.text.tolist(), batch_size=bs, normalize_embeddings=True, show_progress_bar=True)
    np.savez_compressed(out, x=x.astype("float16"), y=df.unsafe.values.astype("int8"),
                        split=np.asarray(df.split.tolist(), dtype="U8"), lang=np.asarray(df.lang.tolist(), dtype="U4"),
                        adv=(df.adv.values if "adv" in df else np.zeros(len(df))).astype("int8"))


if __name__ == "__main__":
    embed(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or []))
