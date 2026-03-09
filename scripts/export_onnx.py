"""Export a Lightning checkpoint to ONNX.

The exported model corresponds to a single forward pass: input image (NCHW) and
auxiliary feature vector (B, 9) -> wind profile (B, 151). Rotation blending
is not included; run it in Python if needed.

Usage:
  # From repo root, with config (ckpt_path required):
  python scripts/export_onnx.py ckpt_path=path/to/model.ckpt

  # Optional: set output path and input shape for fixed-size export
  python scripts/export_onnx.py ckpt_path=path/to/model.ckpt export.output_file=model.onnx \\
    export.image_height=180 export.image_width=103

  # Or via env (for paths with '=' in the name):
  CKPT_PATH=path/to/epoch=29-step=2280.ckpt python scripts/export_onnx.py
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dsat_lightning.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs")


def export_onnx(cfg: DictConfig) -> None:
    """Load Lightning checkpoint and export the model to ONNX."""
    from omegaconf.base import ContainerMetadata
    from omegaconf import ListConfig, OmegaConf as OmegaConfModule

    torch.serialization.add_safe_globals(
        [
            ListConfig,
            OmegaConfModule,
            ContainerMetadata,
        ]
    )

    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError(
            "ckpt_path is required. Set via CLI (ckpt_path=path/to.ckpt) or env: CKPT_PATH=path/to.ckpt"
        )

    export_cfg = cfg.get("export") or {}
    out_file = export_cfg.get("output_file", "model.onnx")
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_file

    # Optional fixed input size (default: dynamic spatial dims)
    batch_size = export_cfg.get("batch_size", 1)
    image_height = export_cfg.get("image_height")
    image_width = export_cfg.get("image_width")
    num_channels = export_cfg.get("num_channels", 1)
    num_features = 9

    log.info("Instantiating model from config...")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Input placeholders: image (B, C, H, W), feature (B, 9)
    if image_height is not None and image_width is not None:
        dummy_image = torch.randn(batch_size, num_channels, image_height, image_width)
        dummy_feature = torch.randn(batch_size, num_features)
        dynamic_axes = None
    else:
        dummy_image = torch.randn(batch_size, num_channels, 180, 103)
        dummy_feature = torch.randn(batch_size, num_features)
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "feature": {0: "batch_size"},
            "profile": {0: "batch_size"},
        }

    log.info("Exporting to ONNX: %s", out_path)
    torch.onnx.export(
        model,
        (dummy_image, dummy_feature),
        str(out_path),
        input_names=["image", "feature"],
        output_names=["profile"],
        dynamic_axes=dynamic_axes,
        opset_version=export_cfg.get("opset_version", 18),
        do_constant_folding=True,
        dynamo=False,
    )
    log.info("Saved ONNX model to %s", out_path)


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="export_onnx.yaml")
def main(cfg: DictConfig) -> None:
    if os.environ.get("CKPT_PATH"):
        cfg.ckpt_path = os.environ["CKPT_PATH"]
    OmegaConf.resolve(cfg)
    export_onnx(cfg)


if __name__ == "__main__":
    main()
