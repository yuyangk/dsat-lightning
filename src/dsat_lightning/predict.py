"""Run model on test set and save predicted (and optional GT) profiles."""

import os
import typing
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dsat_lightning.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def predict(cfg: DictConfig) -> None:
    """Load checkpoint, run on test set, save profiles to disk."""
    # Allow Lightning checkpoint contents (PyTorch 2.6+ weights_only=True)
    from omegaconf.base import ContainerMetadata
    from omegaconf import ListConfig, OmegaConf as OmegaConfModule
    torch.serialization.add_safe_globals([
        ListConfig, OmegaConfModule, ContainerMetadata,
        typing.Any,
    ])

    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("predict.ckpt_path is required")

    # Instantiate datamodule and model from config (same as train)
    log.info("Instantiating datamodule and model...")
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Load weights (state_dict only so we can use weights_only=True and avoid OmegaConf)
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    pred_list: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []

    log.info("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            feature = batch["feature"].to(device)
            pred = model._rotation_blending(images, feature)
            pred_list.append(pred.cpu())
            gt_list.append(batch["profile"])

    pred_profiles = torch.cat(pred_list, dim=0)
    gt_profiles = torch.cat(gt_list, dim=0)

    predict_cfg = cfg.get("predict") or {}
    out_path = Path(cfg.paths.output_dir) / predict_cfg.get("output_file", "profiles.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"pred_profiles": pred_profiles, "gt_profiles": gt_profiles},
        out_path,
    )
    log.info("Saved pred_profiles %s and gt_profiles %s to %s", pred_profiles.shape, gt_profiles.shape, out_path)


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Entry point: run prediction and save profiles."""
    # Allow checkpoint path from env (avoids Hydra parsing paths with '=' in filename)
    if os.environ.get("CKPT_PATH"):
        cfg.ckpt_path = os.environ["CKPT_PATH"]
    OmegaConf.resolve(cfg)
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path is required. Set via CLI (ckpt_path=path/to.ckpt) or env: CKPT_PATH=path/to.ckpt"
        )
    predict(cfg)


if __name__ == "__main__":
    main()
