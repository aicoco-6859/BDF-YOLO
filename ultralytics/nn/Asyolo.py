import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, GhostModule, SELayer
from .modules.transformer import TransformerBlock


class GAM(nn.Module):
    """Global Attention Mechanism (GAM) module with channel and spatial attention."""

    def __init__(self, channels, reduction=16):
        """Initialize GAM module with given parameters."""
        super().__init__()
        self.channels = channels

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward propagation through GAM module."""
        # Channel attention
        chn_att = self.channel_attention(x)
        x = x * chn_att

        # Spatial attention
        spa_att = self.spatial_attention(x)
        x = x * spa_att

        return x

class GhostBottleneck(nn.Module):
    """Ghost Bottleneck module with optional Squeeze-and-Excitation layer."""

    def __init__(self, c1, c2, k=3, s=1, act=False):
        """Initialize GhostBottleneck module with given parameters."""
        super().__init__()
        assert s in [1, 2]
        hidden_dim = c1 // 2

        self.conv = nn.Sequential(
            GhostModule(c1, hidden_dim, kernel_size=1, stride=1, act=True),
            DWConv(hidden_dim, hidden_dim, k, s, act=False) if s == 2 else nn.Identity(),
            SELayer(hidden_dim) if act else nn.Identity(),
            GhostModule(hidden_dim, c2, kernel_size=1, stride=1, act=False),
        )

        if s == 1 and c1 == c2:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                DWConv(c1, c1, k, s, act=True),
                nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        """Forward propagation through GhostBottleneck layer with optional skip connection."""
        return self.conv(x) + self.shortcut(x)

