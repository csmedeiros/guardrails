"""Third-party guard: Qwen3Guard-Gen (generative). P(unsafe) from the first-token probabilities of Safe / Controversial / Unsafe
after the model's own "Safety: " prefix, so no text generation is needed. Same interface as GuardModel."""
import numpy as np


class Qwen3Guard:
    def __init__(self, path: str, block_controversial: bool = True, device: str = "cuda", thr: float = 0.5):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.thr, self.bc = torch, thr, block_controversial
        self.dev = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16 if self.dev == "cuda" else torch.float32).to(self.dev).eval()
        self.ids = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Safe", "Controversial", "Unsafe")]
        assert len(set(self.ids)) == 3, "Safe/Controversial/Unsafe must start with different tokens"

    def prob_unsafe(self, texts: list[str], bs: int = 8) -> np.ndarray:
        torch, out = self.torch, []
        for i in range(0, len(texts), bs):
            prompts = [self.tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False) + "Safety: " for t in texts[i:i + bs]]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.dev)
            with torch.no_grad():
                logits = self.m(**enc).logits[:, -1, self.ids].float()
            p = torch.softmax(logits, -1).cpu().numpy()  # columns: safe, controversial, unsafe
            out.append(p[:, 2] + (p[:, 1] if self.bc else 0.0))
        return np.concatenate(out).astype("float32")

    def classify(self, texts: list[str]) -> list[dict]:
        return [{"unsafe_prob": float(p), "label": "unsafe" if p >= self.thr else "safe", "threshold": self.thr} for p in self.prob_unsafe(texts)]
