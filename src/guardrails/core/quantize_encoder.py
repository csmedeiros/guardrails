"""One-off: 8-bit quantize the frozen bge-m3 encoder (weight-only int8 Linear + int8 embedding tables) -> models/encoder_int8.pt (CPU only).
run: PYTHONPATH=src uv run python src/guardrails/core/quantize_encoder.py     (needs ~4 GB RAM once; serving then needs ~1/3 of fp32)
Serve it with GUARD_QUANT=int8. The head (models/head*.npz) is untouched and stays fp32 (0.6M params)."""
import os
import sys

import torch
from sentence_transformers import SentenceTransformer
from torch.ao.quantization import float_qparams_weight_only_qconfig, quantize_dynamic

from guardrails.core.int8_linear import quantize_linears

ENCODER = os.environ.get("GUARD_ENCODER", "BAAI/bge-m3")
OUT = sys.argv[1] if len(sys.argv) > 1 else "models/encoder_int8.pt"

enc = SentenceTransformer(ENCODER, device="cpu")
enc.max_seq_length = 256
enc = quantize_linears(enc)  # Linear: per-channel int8 weights, fp32 activations
enc = quantize_dynamic(enc, {torch.nn.Embedding: float_qparams_weight_only_qconfig}, inplace=True)  # embedding tables (256M params): lossless in tests
torch.save(enc, OUT)  # full pickle: load only files you made yourself (GuardModel uses weights_only=False)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 2**20:.0f} MiB)")
