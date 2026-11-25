"""Denoising network and RGB inference helpers."""

from .arch_unet import UNet


def load_denoiser(weights, device):
    """Load a supplied network checkpoint on the requested device."""
    import torch

    model = UNet(in_nc=3, out_nc=3, n_feature=48).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model


def denoise_rgb(model, image, device):
    """Denoise uint8 RGB data, padding to the network's 32-pixel grid."""
    import numpy as np
    import torch
    from torch.nn import functional as F

    height, width = image.shape[:2]
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = F.pad(tensor, (0, (-width) % 32, 0, (-height) % 32), mode='replicate')
    with torch.inference_mode():
        result = model(tensor)[0, :, :height, :width].clamp(0, 1)
    return (result.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


__all__ = ['UNet', 'load_denoiser', 'denoise_rgb']
