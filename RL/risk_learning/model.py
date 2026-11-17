"""Risk prediction accepts pooled features + observable context, never an env."""
import hashlib
import numpy as np
import torch
from torch import nn

FORMAT_VERSION = 1
FEATURE_SPEC = {
    "visual_dim": 512, "context_dim": 9,
    "speed_scale_mps": 10.0, "speed_clip": [0.0, 2.0],
    "target_scale_m": 30.0, "target_clip": [-1.0, 1.0],
    "commands": 6, "command_indexing": "1-based",
    "pooling": "mean of post-GGCA spatial feature before backbone fc",
    "image_normalization": "ImageNet RGB mean/std; no resize",
    "rgb_shape": [256, 900, 3],
}


def observable_context(speed, target_point, command):
    target = np.asarray(target_point, dtype=np.float32)
    if target.shape != (2,) or not np.isfinite(target).all() or not np.isfinite(speed) or speed < 0:
        raise ValueError("Invalid observable speed/navigation")
    if int(command) != command or not 1 <= int(command) <= 6:
        raise ValueError("Navigation command must be an integer in [1, 6]")
    one_hot = np.zeros(6, dtype=np.float32)
    one_hot[int(command) - 1] = 1
    return np.concatenate(([np.clip(speed / 10.0, 0, 2)], np.clip(target / 30.0, -1, 1), one_hot)).astype(np.float32)


class RiskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(521, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 521:
            raise ValueError("Risk head expects a batch of 521-dimensional features")
        return self.layers(features).squeeze(-1)


def encoder_signature(tcp_net):
    digest = hashlib.sha256()
    for name, tensor in sorted(tcp_net.perception.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_head(path, expected_encoder_signature, device):
    # Only load trusted local model files; torch.load uses pickle on PyTorch 1.7.
    payload = torch.load(str(path), map_location="cpu")
    if payload.get("format_version") != FORMAT_VERSION or payload.get("feature_spec") != FEATURE_SPEC:
        raise ValueError("Unsupported risk checkpoint format or feature normalization")
    if payload.get("encoder_signature") != expected_encoder_signature:
        raise ValueError("Risk checkpoint was trained with a different frozen TCP encoder")
    head = RiskHead()
    head.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(p).all().item() for p in head.parameters()):
        raise ValueError("Risk checkpoint contains non-finite parameters")
    head.to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload


def predict_risk(head, features):
    x = np.asarray(features, dtype=np.float32)
    if x.shape != (521,) or not np.isfinite(x).all():
        raise ValueError("Online risk features must be finite and have shape (521,)")
    with torch.no_grad():
        device = next(head.parameters()).device
        value = float(head(torch.from_numpy(x).unsqueeze(0).to(device)).item())
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Non-finite or out-of-range risk prediction; no rule fallback is permitted")
    return value
