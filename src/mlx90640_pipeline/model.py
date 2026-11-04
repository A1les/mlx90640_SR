"""Small temperature-domain x2 CNN. Inputs and outputs are float32 degrees C."""
import torch
from torch import nn
from torch.nn import functional as F


class ThermalSRx2(nn.Module):
    def __init__(self, channels=24, hidden_layers=3, center_input=True):
        super().__init__()
        self.center_input = center_input
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU()]
        for _ in range(hidden_layers):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        # Exactly bilinear at initialization. No per-frame min/max scaling.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, temperature_c):
        base = F.interpolate(temperature_c, scale_factor=2.0, mode='bilinear', align_corners=False)
        center = temperature_c.mean(dim=(-2, -1), keepdim=True) if self.center_input else 25.0
        normalized = (temperature_c - center) / 10.0
        residual_c = self.shuffle(self.head(self.features(normalized))) * 10.0
        return base + residual_c


def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['model_config']
    model = ThermalSRx2(**config)
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(), checkpoint
