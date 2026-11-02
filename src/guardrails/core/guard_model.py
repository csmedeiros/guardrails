"""Inference wrapper: frozen bge-m3 encoder + trained MLP head (plain numpy weights) -> P(unsafe)."""
import os

import numpy as np

HEAD = os.environ.get("GUARD_HEAD", "models/head.npz")
ENCODER = os.environ.get("GUARD_ENCODER", "BAAI/bge-m3")
DEVICE = os.environ.get("GUARD_DEVICE", "cpu")  # cpu keeps the GPU free for the serving LLM; set cuda for speed
MODEL_DIR = os.environ.get("GUARD_MODEL_DIR", "")  # single fine-tuned model (guard_net.py); overrides head + encoder
QUANT = os.environ.get("GUARD_QUANT", "")  # "int8": 8-bit encoder from ENCODER_INT8 (CPU only); "" = fp32 (default)
ENCODER_INT8 = os.environ.get("GUARD_ENCODER_INT8", "models/encoder_int8.pt")  # made by quantize_encoder.py


class GuardModel:
    def __init__(self, head: str = HEAD, encoder: str = ENCODER, device: str = DEVICE, quant: str = QUANT, model_dir: str = MODEL_DIR):
        import torch

        self.net = None
        if model_dir:
            from guardrails.core.guard_net import GuardNet, tokenizer

            self.net, self.tok, self.device = GuardNet.load(model_dir), tokenizer(), device
            self.thr = self.net.thr
            return
        z = np.load(head)
        self.thr = float(z["thr"])
        n = sum(k.startswith("w") for k in z.files)
        self.layers = [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
        if quant == "int8":
            if device != "cpu":
                raise ValueError("GUARD_QUANT=int8 is CPU-only; use GUARD_DEVICE=cpu")
            # Own file written by quantize_encoder.py (a full pickle, hence weights_only=False). Never loads the fp32 checkpoint.
            self.enc = torch.load(ENCODER_INT8, map_location="cpu", weights_only=False)
        elif quant:
            raise ValueError(f"unknown GUARD_QUANT={quant!r}; use 'int8' or leave empty")
        else:
            from sentence_transformers import SentenceTransformer

            kw = {"model_kwargs": {"torch_dtype": torch.float16}} if device == "cuda" else {}
            self.enc = SentenceTransformer(encoder, device=device, **kw)
        self.enc.max_seq_length = 256

    def prob_unsafe(self, texts: list[str]) -> np.ndarray:
        if self.net is not None:
            return self.net.prob(texts, self.tok, device=self.device)
        h = self.enc.encode(texts, batch_size=32, normalize_embeddings=True).astype("float32")
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return (1 / (1 + np.exp(-h[:, 0]))).astype("float32")

    def classify(self, texts: list[str]) -> list[dict]:
        return [{"unsafe_prob": float(p), "label": "unsafe" if p >= self.thr else "safe", "threshold": self.thr}
                for p in self.prob_unsafe(texts)]
