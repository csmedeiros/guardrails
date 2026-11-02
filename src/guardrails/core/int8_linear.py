"""Weight-only int8 Linear: per-output-channel symmetric int8 weights, fp32 activations (dequantized per call).
Dynamic int8 (quantized activations) badly hurt bge-m3 (activation outliers: only ~80% label agreement with fp32)."""
import torch


class Int8Linear(torch.nn.Module):
    def __init__(self, lin: torch.nn.Linear):
        super().__init__()
        w = lin.weight.data
        scale = (w.abs().amax(dim=1) / 127).clamp_min(1e-12)
        self.register_buffer("qweight", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.float())
        self.register_buffer("bias", None if lin.bias is None else lin.bias.data.clone())

    def forward(self, x):
        return torch.nn.functional.linear(x, self.qweight.to(x.dtype) * self.scale[:, None].to(x.dtype), self.bias)


def quantize_linears(model: torch.nn.Module) -> torch.nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(model, name, Int8Linear(child))
        else:
            quantize_linears(child)
    return model
