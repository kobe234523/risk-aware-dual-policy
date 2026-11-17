"""Reuse one TCP forward pass for actions and the post-GGCA pooled feature."""
import numpy as np
import torch
from .model import observable_context


def tcp_actions_and_features(net, rgb, speed, target_point, command):
    if net.training or net.perception.training:
        raise ValueError("TCP and its encoder must be in evaluation mode")
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.shape != (256, 900, 3):
        raise ValueError("Expected 256 x 900 x 3 uint8 RGB, matching the TCP camera")
    context = observable_context(speed, target_point, command)
    device = next(net.parameters()).device
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(device).float() / 255.0
    mean = rgb_tensor.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = rgb_tensor.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_tensor = ((rgb_tensor - mean) / std).unsqueeze(0)
    # TCP keeps its original speed/12 normalization; risk context independently uses speed/10.
    speed_tensor = torch.tensor([[speed / 12.0]], dtype=torch.float32, device=device)
    target_tensor = torch.tensor([target_point], dtype=torch.float32, device=device)
    cmd = torch.from_numpy(context[3:]).view(1, 6).to(device)
    state = torch.cat([speed_tensor, target_tensor, cmd], dim=1)
    captured = []

    def capture(_module, _inputs, output):
        spatial = output[1]
        if spatial.ndim != 4 or spatial.shape[0] != 1 or spatial.shape[1] != 512:
            raise ValueError("Expected a post-GGCA ResNet feature map with 512 channels")
        captured.append(spatial.detach().mean(dim=(2, 3)).squeeze(0).cpu().numpy())

    handle = net.perception.register_forward_hook(capture)
    try:
        with torch.no_grad():
            pred = net(rgb_tensor, state, target_tensor)
            ctrl = net.process_action(pred, command, speed_tensor, target_tensor)[:3]
            traj = net.control_pid(pred["pred_wp"], speed_tensor, target_tensor)[:3]
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Expected exactly one visual encoder forward pass")

    def action(values):
        steer, throttle, brake = [float(np.asarray(v).item()) for v in values]
        return np.array([steer, throttle if throttle > 0 else -brake], dtype=np.float32)

    features = np.concatenate([captured[0], context]).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite encoder features")
    return action(traj), action(ctrl), features
