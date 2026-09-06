"""Generator architecture for Traj-AdvGAN."""

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            nn.BatchNorm2d(dim),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ChannelAttention(nn.Module):
    """SE-style channel attention used after the residual bottleneck.

    GAP -> FC reduction -> ReLU -> FC expansion -> Sigmoid -> channel reweighting.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class TrafficSignGenerator(nn.Module):
    """Encoder -> residual bottleneck -> channel attention -> decoder.

    The final tanh output is in (-1, 1). The attack class multiplies it by epsilon.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        attention_reduction: int = 4,
        use_attention: bool = True,
        num_resblocks: int = 4,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(*[ResnetBlock(256) for _ in range(num_resblocks)])
        self.channel_attention = (
            ChannelAttention(256, reduction=attention_reduction)
            if use_attention
            else nn.Identity()
        )
        # Bilinear upsampling + convolution avoids transposed-convolution checkerboard artifacts.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, output_nc, kernel_size=3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.bottleneck(h)
        h = self.channel_attention(h)
        return self.decoder(h)
