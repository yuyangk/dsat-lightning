"""In-memory PyTorch Dataset for TCSA tropical-cyclone satellite data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from dsat_lightning.dataset.image_processor import ImageProcessor


def random_rotation(tensor_images: torch.Tensor) -> torch.Tensor:
    images_np = tensor_images.cpu().numpy()
    images_np = ImageProcessor(images_np).random_rotate()
    return torch.as_tensor(
        images_np, dtype=tensor_images.dtype, device=tensor_images.device
    )


class TCSADataset(Dataset):
    """In-memory dataset for TCSA tropical-cyclone satellite samples.

    Each item is a dict with keys:

    * ``image``    (H, W, 4)  float32  IR1 / WV / VIS / PMW
    * ``feature``  (9,)       float32  auxiliary features
    * ``profile``  (151,)     float32  wind-speed profile (0–750 km, 5 km steps)
    * ``vmax``     (1,)       float32  max sustained wind (kt)
    * ``r34``      (1,)       float32  radius of 34-kt wind (km)
    * ``random_rotation_function`` (Callable[[torch.Tensor], torch.Tensor])  function to apply random rotation to the image
    """

    def __init__(
        self,
        images: np.ndarray,
        features: np.ndarray,
        profiles: np.ndarray,
        vmaxs: np.ndarray,
        r34s: np.ndarray,
        is_random_rotation: bool = False,
    ) -> None:
        self.images = torch.as_tensor(images, dtype=torch.float32)
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.profiles = torch.as_tensor(profiles, dtype=torch.float32)
        self.vmaxs = torch.as_tensor(vmaxs, dtype=torch.float32)
        self.r34s = torch.as_tensor(r34s, dtype=torch.float32)
        self.is_random_rotation = is_random_rotation

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = self.images[idx]  # (H, W, 4)
        if self.is_random_rotation:
            image = random_rotation(image)

        return {
            "image": image,
            "feature": self.features[idx],
            "profile": self.profiles[idx],
            "vmax": self.vmaxs[idx],
            "r34": self.r34s[idx],
        }
