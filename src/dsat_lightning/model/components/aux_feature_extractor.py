"""Auxiliary feature extraction from raw feature vectors."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxFeatureExtractor(nn.Module):
    """Extract region one-hot + cyclic time features from the raw feature vector.

    Feature columns: ``[lon, lat, region_code, yday_cos, yday_sin,
    hour_cos, hour_sin, minutes_to_noon, is_good_VIS]``

    Output: ``(B, num_regions + 4)``  (one-hot region + 4 time features).
    """

    def __init__(self, num_regions: int = 6) -> None:
        super().__init__()
        self.num_regions = num_regions

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        region = F.one_hot(feature[:, 2].long(), self.num_regions).float()
        time_feats = feature[:, 3:7]  # yday_cos, yday_sin, hour_cos, hour_sin
        return torch.cat([region, time_feats], dim=1)
