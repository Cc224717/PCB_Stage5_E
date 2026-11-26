import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, c1=None, c2=None, k=1, s=1, p=None, g=1, d=1, act=True, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.channel_attention = None
        self.spatial_attention = SpatialAttention()
        self.align = nn.Identity()

    def _build(self, channels, device):
        self.channel_attention = ChannelAttention(channels, self.reduction).to(device)

    def forward(self, x):
        if self.channel_attention is None:
            self._build(x.shape[1], x.device)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return self.align(x)
