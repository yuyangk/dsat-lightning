"""Lightning module wrapping the Profiler with training / eval logic."""

from typing import Sequence

import torch
import torch.nn as nn
import lightning as L
from einops import rearrange

from .profiler import Profiler
from .utils import profiles_to_metrics
from dsat_lightning.dataset.image_processor import ImageProcessor


class ProfilerModule(L.LightningModule):
    """Lightning wrapper: multi-task loss, rotation augmentation, blending eval.

    Args:
        input_channels: Channel indices for the backbone.
        num_regions: Number of basin regions.
        loss_function: ``'MAE'`` or ``'MSE'``.
        loss_ratio: Weight per loss term,
            e.g. ``{'Vmax': 0.125, 'profile': 1.0, 'RMW': 0.0125}``.
        blending_num: Evenly-spaced rotations for validation blending.
        lr: Adam learning rate.
        vmax_loss_sample_weight_exponent:
            Raise GT Vmax to this power for sample weighting (0 = uniform).
    """

    def __init__(
        self,
        input_channels: Sequence[int] = (0,),
        num_regions: int = 6,
        loss_function: str = "MAE",
        loss_ratio: dict[str, float] = {"Vmax": 1.0, "profile": 1.0, "RMW": 0.0},
        blending_num: int = 10,
        lr: float = 1e-3,
        vmax_loss_sample_weight_exponent: float = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.net = Profiler(input_channels=input_channels, num_regions=num_regions)
        self.loss_fn: nn.Module = (
            nn.L1Loss(reduction="none")
            if loss_function == "MAE"
            else nn.MSELoss(reduction="none")
        )

    def forward(self, image: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return self.net(image, feature)

    def _prepare_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return rearrange(images, "b h w c -> b c h w")

    @torch.no_grad()
    def _rotation_blending(
        self,
        images: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        evenly_rotated_images = ImageProcessor(images.cpu().numpy()).evenly_rotate(
            self.hparams.blending_num
        )
        preds: list[torch.Tensor] = []
        for image in evenly_rotated_images:
            image = torch.as_tensor(image, dtype=images.dtype, device=images.device)
            image = rearrange(image, "b h w c -> b c h w")
            pred = self(image, feature)
            preds.append(pred)
        return torch.stack(preds).mean(dim=0)  # (B, 151)

    # ---- loss ------------------------------------------------------------

    def _compute_losses(
        self,
        pred_profile: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Multi-task losses derived from the predicted wind profile.

        Loss terms mirror the original TF training code:
        * **Vmax** – MAE/MSE between profile-derived Vmax on both sides.
        * **profile** – element-wise loss masked for invalid profiles (sentinel -1).
        * **RMW** – MAE/MSE between profile-derived radius of max wind.
        """
        gt_profile = batch["profile"]
        pred_vmax, _, pred_rmw = profiles_to_metrics(pred_profile)
        gt_vmax, _, gt_rmw = profiles_to_metrics(gt_profile)

        # Vmax loss (optionally sample-weighted by magnitude)
        vmax_loss = self.loss_fn(pred_vmax, gt_vmax).squeeze(1)  # (B,)
        if self.hparams.vmax_loss_sample_weight_exponent > 0:
            w = gt_vmax.squeeze(1) ** self.hparams.vmax_loss_sample_weight_exponent
            vmax_loss = vmax_loss * (w / w.mean())

        profile_loss = self.loss_fn(pred_profile, gt_profile).mean(1)  # (B,)

        # RMW loss
        rmw_loss = self.loss_fn(pred_rmw, gt_rmw).squeeze(1)  # (B,)

        return {
            "Vmax": vmax_loss.mean(),
            "profile": profile_loss.mean(),
            "RMW": rmw_loss.mean(),
        }

    def _weighted_total(
        self,
        losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Total loss = sum over terms in config loss_ratio (e.g. 0.125*Vmax + 1.0*profile + 0.0125*RMW)
        return sum(v * self.hparams.loss_ratio.get(k, 0.0) for k, v in losses.items())

    # ---- Lightning steps -------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        images = self._prepare_images(batch["image"])
        pred = self(images, batch["feature"])
        losses = self._compute_losses(pred, batch)
        total = self._weighted_total(losses)

        self.log_dict({f"train/{k}_loss": v for k, v in losses.items()})
        self.log("train/total_loss", total, prog_bar=True)
        return total

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        pred = self._rotation_blending(batch["image"], batch["feature"])
        losses = self._compute_losses(pred, batch)
        total = self._weighted_total(losses)

        self.log_dict(
            {f"val/{k}_loss": v for k, v in losses.items()},
            sync_dist=True,
        )
        self.log("val/total_loss", total, prog_bar=True, sync_dist=True)
        return total

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        pred = self._rotation_blending(batch["image"], batch["feature"])
        losses = self._compute_losses(pred, batch)
        total = self._weighted_total(losses)

        self.log_dict(
            {f"test/{k}_loss": v for k, v in losses.items()},
            sync_dist=True,
        )
        self.log("test/total_loss", total, sync_dist=True)
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
