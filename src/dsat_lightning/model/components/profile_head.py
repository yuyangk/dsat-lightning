"""FC head for profile prediction."""

import torch
import torch.nn as nn


class ProfileHead(nn.Module):
    """FC head that maps flattened visual + auxiliary features to a profile.

    Uses ``LazyLinear`` so the input dimension adapts automatically to different
    spatial sizes (polar 180x103 vs cartesian 64x64).

    Args:
        hidden_dims: Sizes of intermediate FC layers.
        output_dim:  Output dimension (151 for 0–750 km at 5 km steps).
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 64),
        output_dim: int = 151,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for dim in hidden_dims:
            layers += [nn.LazyLinear(dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True)]
        layers.append(nn.LazyLinear(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) concatenated visual + auxiliary features.

        Returns:
            (B, output_dim) predicted profile.
        """
        return self.net(x)
