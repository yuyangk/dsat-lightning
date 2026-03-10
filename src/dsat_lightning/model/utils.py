"""Pure tensor utility functions (no trainable parameters)."""

import torch


def profiles_to_metrics(
    profiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive Vmax, R34, RMW from a 151-element wind-speed profile.

    Args:
        profiles: (B, 151) wind speed at 0–750 km in 5 km steps.

    Returns:
        vmax: (B, 1) maximum wind speed (kt).
        r34:  (B, 1) outermost radius of 34-kt wind (km).
        rmw:  (B, 1) radius of maximum wind (km).
    """
    vmax = profiles.max(dim=1, keepdim=True).values

    radii = torch.arange(0, 751, 5, dtype=profiles.dtype, device=profiles.device)
    r34 = ((profiles >= 34.0).float() * radii).max(dim=1, keepdim=True).values

    # weights = torch.softmax(profiles, dim=1)
    # rmw = (weights * radii).sum(dim=1, keepdim=True)
    rmw = radii[profiles.argmax(dim=1, keepdim=True)]

    return vmax, r34, rmw
