"""Profiler CNN backbone for tropical-cyclone wind-profile prediction."""

from typing import Sequence

import torch
import torch.nn as nn

from .components import Conv2d, AuxFeatureExtractor, ProfileHead


class Profiler(nn.Module):
    """CNN for tropical-cyclone wind-profile prediction.

    Architecture mirrors the TF ``profiler_4_x`` family:

    * 6 conv layers (kernel 4x3, stride 2, same-padding) + BN + ReLU
    * Flatten -> concat with auxiliary features (region one-hot + time)
    * 2 FC layers with BN + ReLU -> 151-dim output profile

    Args:
        input_channels: Image channel indices to use.
            ``(0,)`` = IR1 only;  ``(0, 2)`` = IR1 + VIS.
        num_regions: Number of basin regions for one-hot encoding.
    """

    def __init__(
        self,
        input_channels: Sequence[int] = (0,),
        num_regions: int = 6,
    ) -> None:
        super().__init__()
        self.input_channels = list(input_channels)
        n_in = len(input_channels)

        # Input normalisation
        self.input_norm = nn.BatchNorm2d(n_in)

        # 6-layer conv backbone
        ch = [n_in, 16, 32, 64, 128, 256, 512]
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Conv2d(ch[i], ch[i + 1], kernel_size=(4, 3), stride=2),
                    nn.BatchNorm2d(ch[i + 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(6)
            ]
        )

        # Auxiliary feature extractor
        self.aux_extractor = AuxFeatureExtractor(num_regions=num_regions)

        # FC head  (LazyLinear adapts to spatial dims: polar 180x103 vs cart 64x64)
        self.fc_head = ProfileHead(hidden_dims=(256, 64), output_dim=151)

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:   (B, C, H, W)  satellite imagery in **NCHW** format.
            feature: (B, 9)        auxiliary feature vector.

        Returns:
            (B, 151) predicted wind-speed profile.
        """
        x = image[:, self.input_channels, :, :]
        x = self.input_norm(x)

        for block in self.conv_blocks:
            x = block(x)

        x = x.flatten(1)  # (B, 512*h*w)
        x = torch.cat([x, self.aux_extractor(feature)], dim=1)
        x = self.fc_head(x)
        return x
