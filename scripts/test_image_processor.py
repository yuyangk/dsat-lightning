from dsat_lightning.dataset.image_processor import ImageProcessor
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_single_image_from_h5(h5_path: str, index: int = 0) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        image = f["images"][index, :, :, :]
        image = np.squeeze(image)
    return image


def load_batch_image_from_h5(
    h5_path: str, index: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        images = f["images"][index]
        images = np.squeeze(images)
    return images




def cart_test(file_path: str):
    image = load_single_image_from_h5(file_path)
    image_processor = ImageProcessor(image)
    plot_image(image_processor.evenly_rotate(5), "single_cart_evenly_rotate")
    plot_image(image_processor.random_rotate(), "single_cart_random_rotate")
    plot_image(image_processor.flip_images(), "single_cart_flip_images")
    plot_image(image_processor.crop_center(64), "single_cart_crop_center")
    plot_image(image_processor.cart2polar(), "single_cart_cart2polar")

def polar_test(file_path: str):
    image = load_single_image_from_h5(file_path)
    image_processor = ImageProcessor(image)
    plot_image(image_processor.cart2polar(), "single_polar_cart2polar")
    plot_image(image_processor.evenly_rotate(5), "single_polar_evenly_rotate")
    plot_image(image_processor.random_rotate(), "single_polar_random_rotate")


def batch_cart_test(file_path: str):
    images = load_batch_image_from_h5(file_path)
    image_processor = ImageProcessor(images)
    plot_image(image_processor.evenly_rotate(5), "batch_cart_evenly_rotate")
    plot_image(image_processor.random_rotate(), "batch_cart_random_rotate")
    plot_image(image_processor.flip_images(), "batch_cart_flip_images")
    plot_image(image_processor.crop_center(64), "batch_cart_crop_center")
    plot_image(image_processor.cart2polar(), "batch_cart_cart2polar")


def batch_polar_test(file_path: str):
    images = load_batch_image_from_h5(file_path)
    image_processor = ImageProcessor(images)
    plot_image(image_processor.cart2polar(), "batch_polar_cart2polar")
    plot_image(image_processor.evenly_rotate(5), "batch_polar_evenly_rotate")
    plot_image(image_processor.random_rotate(), "batch_polar_random_rotate")
    plot_image(image_processor.polar2cart(), "batch_polar_polar2cart")

def plot_image(images: np.ndarray, title: str = ""):
    n_rot = None  # set when input is 5D (num_rotations, batch, h, w, c)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)
    if images.ndim == 5:
        n_rot, batch, h, w, c = images.shape
        # (num_rotations, batch, h, w, c) -> (num_rotations * batch, h, w, c)
        images = images.reshape(n_rot * batch, h, w, c)

    # plot all images in the batch and channel dimension in a grid
    print(images.shape)
    batch_size, h, w, channel_size = images.shape
    for i in range(batch_size):
        for j in range(channel_size):
            plt.subplot(batch_size, channel_size, i * channel_size + j + 1)
            plt.imshow(images[i, :, :, j], cmap="gray")
            plt.axis("off")
            # label rows by rotation when we had 5D input
            if n_rot is not None and j == 0:
                rot_idx = i % n_rot
                plt.ylabel(f"rot {rot_idx}", fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    # get the folder of current file
    current_file_path = Path(__file__).resolve().parent
    plt.savefig(f"{current_file_path}/images/{title}.png")
    plt.close()


def main():
    file_path = "data/TCSA_2004_2018.h5"
    cart_test(file_path)
    polar_test(file_path)
    batch_cart_test(file_path)
    batch_polar_test(file_path)


if __name__ == "__main__":
    main()
