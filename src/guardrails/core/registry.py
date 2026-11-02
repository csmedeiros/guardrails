"""Pick the guard model from configs/guard_models.yaml: GUARD_MODEL=<name> (env) or the file's `default`.
Without GUARD_MODEL the old env vars (GUARD_MODEL_DIR / GUARD_HEAD) still work through GuardModel."""
import os

import yaml

CONFIG = os.environ.get("GUARD_MODELS_CONFIG", "configs/guard_models.yaml")


def available() -> dict:
    return yaml.safe_load(open(CONFIG))["models"]


def load_guard(name: str | None = None):
    cfg = yaml.safe_load(open(CONFIG))
    legacy = os.environ.get("GUARD_MODEL_DIR") or os.environ.get("GUARD_HEAD")
    name = name or os.environ.get("GUARD_MODEL") or (None if legacy else cfg["default"])
    if name is None:  # legacy env vars, no registry entry chosen
        from guardrails.core.guard_model import GuardModel
        return GuardModel()
    if name not in cfg["models"]:
        raise ValueError(f"unknown guard model {name!r}; choose one of {sorted(cfg['models'])}")
    c, device = cfg["models"][name], os.environ.get("GUARD_DEVICE", "cpu")
    if c["type"] == "qwen3guard":
        from guardrails.core.qwen3guard import Qwen3Guard
        m = Qwen3Guard(c["path"], c.get("block_controversial", True), os.environ.get("GUARD_DEVICE", "cuda"), c.get("threshold", 0.5))
    elif c["type"] == "guardnet":
        from guardrails.core.guard_model import GuardModel
        m = GuardModel(model_dir=c["path"], device=device)
    elif c["type"] == "head":
        from guardrails.core.guard_model import GuardModel
        m = GuardModel(head=c["path"], device=device)
    else:
        raise ValueError(f"{name}: unknown type {c['type']!r}")
    if "threshold" in c:
        m.thr = float(c["threshold"])
    m.name = name
    return m
