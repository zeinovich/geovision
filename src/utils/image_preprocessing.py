import cv2
import numpy as np


def image_processing(
    img: np.ndarray, kernel_size: int = 3, iterations: int = 1
) -> np.ndarray:
    """
    `img`: np.ndarray
        Image
    `kernel_size`: int, default=3
        Size of kernel for morphological operation (`cv2.MORPH_CLOSE`)
    `iterations`: int, default=1
        Number of iterations

    Return
        `np.ndarray` (img)
    """
    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel=kernel,
        iterations=iterations,
    )

    return img
