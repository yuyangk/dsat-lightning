import numpy as np
import cv2

import polarTransform as pt
from dataclasses import dataclass


def _ensure_batch(images: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize to (N, H, W, C). Return (images_4d, was_single)."""
    if images.ndim == 3:
        return np.expand_dims(images, axis=0), True
    if images.ndim == 4:
        return images, False
    raise ValueError(
        f"Expected 3D (H,W,C) or 4D (N,H,W,C) array, got shape {images.shape}"
    )


class ImageProcessor:
    """Process single or batched satellite images (H,W,C) or (N,H,W,C)."""

    def __init__(self, images: np.ndarray):
        self._images_4d, self._single_image = _ensure_batch(images)
        n, h, w, c = self._images_4d.shape
        self.images = self._images_4d
        self.batch_size = n
        self.height = h
        self.width = w
        self.channel_num = c
        self.is_polar = self._is_polar_coordinate()
        self.processor = self._get_processor()

    def _is_polar_coordinate(self) -> bool:
        return self.height != self.width

    def _get_processor(self):
        if self.is_polar:
            return PolarSatelliteImage(self.images)
        return CartesianSatelliteImage(self.images)

    def _maybe_squeeze(self, out: np.ndarray) -> np.ndarray:
        """Return 3D if input was single image, else 4D."""
        if self._single_image and out.shape[0] == 1:
            return out[0]
        return out

    def remove_outlier_and_nan(self, upper_bound: int = 1000) -> np.ndarray:
        cleaned = np.nan_to_num(self.images, copy=False)
        cleaned[cleaned > upper_bound] = 0
        return self._maybe_squeeze(cleaned)

    def flip_images(self) -> np.ndarray:
        return self._maybe_squeeze(self.processor.flip_images())

    def evenly_rotate(self, rotate_num: int) -> np.ndarray:
        rotated = self.processor.evenly_rotate(rotate_num)
        rotated = np.stack(rotated, axis=0)
        return rotated

    def random_rotate(self) -> np.ndarray:
        return self._maybe_squeeze(self.processor.random_rotate())

    def cart2polar(self) -> np.ndarray:
        converted = self.processor.cart2polar()
        self._update_processor_after_conversion(converted)
        return self._maybe_squeeze(converted)

    def polar2cart(self) -> np.ndarray:
        converted = self.processor.polar2cart()
        self._update_processor_after_conversion(converted)
        return self._maybe_squeeze(converted)

    def crop_center(self, crop_width: int) -> np.ndarray:
        return self._maybe_squeeze(self.processor.crop_center(crop_width))

    def _update_processor_after_conversion(self, converted_images: np.ndarray) -> None:
        """Update processor after coordinate system conversion."""
        converted_4d, _ = _ensure_batch(converted_images)
        self.images = converted_4d
        self._images_4d = converted_4d
        n, h, w, c = converted_4d.shape
        self.batch_size = n
        self.height = h
        self.width = w
        self.channel_num = c
        self.is_polar = self._is_polar_coordinate()
        self.processor = self._get_processor()


@dataclass
class BaseSatelliteImage:
    """Base for satellite images. images: (N, H, W, C), N >= 1."""

    images: np.ndarray

    def __post_init__(self) -> None:
        if self.images.ndim == 3:
            self.images = np.expand_dims(self.images, axis=0)
        if self.images.ndim != 4:
            raise ValueError(f"Expected 3D or 4D images, got shape {self.images.shape}")
        self.batch_size, self.height, self.width, self.channel_num = self.images.shape


@dataclass
class CartesianSatelliteImage(BaseSatelliteImage):
    """Satellite images in Cartesian coordinates (square)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.height != self.width:
            raise ValueError(
                "Cartesian Satellite images must be square (height == width)"
            )

    def __repr__(self) -> str:
        return (
            f"CartesianSatelliteImage(batch_size={self.batch_size}, "
            f"height={self.height}, width={self.width}, channel_num={self.channel_num})"
        )

    def _rotate_one(self, img: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a single image (H, W, C) by angle in degrees."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    def _rotate(self, angle: float) -> np.ndarray:
        """Rotate all images by the same angle (degrees). Shape (N, H, W, C)."""
        out = np.stack(
            [self._rotate_one(self.images[i], angle) for i in range(self.batch_size)],
            axis=0,
        )
        return out

    def flip_images(self) -> np.ndarray:
        """Flip images horizontally (axis=2)."""
        return np.flip(self.images, axis=2)

    def crop_center(self, crop_width: int) -> np.ndarray:
        start = self.width // 2 - crop_width // 2
        end = start + crop_width
        return self.images[:, start:end, start:end, :]

    def random_rotate(self) -> np.ndarray:
        """Randomly rotate each image (different angle per image)."""
        angles = np.random.uniform(0, 360, size=self.batch_size)
        out = np.stack(
            [
                self._rotate_one(self.images[i], float(angles[i]))
                for i in range(self.batch_size)
            ],
            axis=0,
        )
        return out

    def evenly_rotate(self, rotate_num: int) -> list[np.ndarray]:
        """Evenly rotate: list of (N, H, W, C) arrays."""
        angles = np.arange(0, 360, 360.0 / rotate_num)
        return [self._rotate(angle) for angle in angles]

    def cart2polar(
        self,
        final_radius: int = 64,
        radius_size: int = 103,
        angle_size: int = 180,
    ) -> np.ndarray:
        """Convert batch to polar. 128x128x? -> 180x103x?."""
        out = []
        for i in range(self.batch_size):
            polar = pt.convertToPolarImage(
                self.images[i],
                hasColor=True,
                finalRadius=final_radius,
                radiusSize=radius_size,
                angleSize=angle_size,
            )[0]
            out.append(polar)
        return np.stack(out, axis=0)


@dataclass
class PolarSatelliteImage(BaseSatelliteImage):
    """Satellite images in polar coordinates (rectangular)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.height == self.width:
            raise ValueError(
                "Polar Satellite images must be rectangular (height != width)"
            )

    def __repr__(self) -> str:
        return (
            f"PolarSatelliteImage(batch_size={self.batch_size}, "
            f"height={self.height}, width={self.width}, channel_num={self.channel_num})"
        )

    def _shift(self, shift_distance: float) -> np.ndarray:
        """Shift all images along angular dimension (axis=1)."""
        return np.roll(self.images, shift=int(round(shift_distance)), axis=1)

    def flip_images(self) -> np.ndarray:
        """Flip along width (axis=2)."""
        return np.flip(self.images, axis=2)

    def evenly_rotate(self, rotate_num: int) -> list[np.ndarray]:
        shift_distances = np.arange(0, self.height, self.height / rotate_num)
        return [self._shift(float(d)) for d in shift_distances]

    def random_rotate(self) -> np.ndarray:
        """Random shift per image along angular dimension."""
        shifts = np.random.uniform(0, self.height, size=self.batch_size)
        out = np.empty_like(self.images)
        for i in range(self.batch_size):
            out[i] = self._shift(shifts[i])[i]
        return np.stack(out, axis=0)

    def polar2cart(
        self,
        final_radius: int = 64,
        image_size: tuple[int, int] = (128, 128),
    ) -> np.ndarray:
        """Convert batch to Cartesian. 180x103x? -> 128x128x?."""
        out = []
        for i in range(self.batch_size):
            cart = pt.convertToCartesianImage(
                self.images[i],
                hasColor=True,
                finalRadius=final_radius,
                imageSize=image_size,
            )[0]
            out.append(cart)
        return np.stack(out, axis=0)
