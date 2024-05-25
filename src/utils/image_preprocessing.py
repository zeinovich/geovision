import cv2
import numpy as np

from paddleocr import PaddleOCR

import math

from typing import List, Tuple


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

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img[gray_img < 200] = 0
    gray_img[gray_img >= 200] = 255

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel=kernel,
        iterations=iterations,
    )

    return img


def get_ocr(
    ocr_model: PaddleOCR, img: np.ndarray, step: int = 500
) -> Tuple[List[List[int]], List[str]]:
    """
    `img`: np.ndarray
        Input image
    `step`: int, default=500
        Size of horizontal crop step

    Returns
    `predictions`: Tuple[List[List[int]], List[str]]
        Tuple of `List` of bboxes and `List` of text
    """
    img = image_processing(img)

    height = img.shape[0]
    step = step
    n_steps = math.ceil(height / step)

    boxes_list = []
    texts_list = []

    for i in range(n_steps):
        bottom = max(0, i * step - 100)
        top = (i + 1) * step

        pred = ocr_model.ocr(img[bottom:top])

        boxes = [[[box[0], box[1] + bottom] for box in p[0]] for p in pred[0]]
        texts = [p[1][0] for p in pred[0]]

        boxes_list.extend(boxes)
        texts_list.extend(texts)

    return boxes_list, texts_list


def cover_detections(img: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
    """
    `img`: np.ndarray
        Original Image
    `boxes`: List[List[int]]
        OCR detections bboxes

    Returns
    `img`: np.ndarray
        Image with covered detections
    """
    img_copy = img.copy()
    mean_color = img_copy.mean(axis=0).mean(axis=0).astype(np.uint8).tolist()

    NARROW = 1

    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[2]
        x1, y1 = int(x1 - NARROW), int(y1 - NARROW)
        x2, y2 = int(x2 - NARROW), int(y2 - NARROW)

        cv2.rectangle(
            img_copy,
            (x1, y1),
            (x2, y2),
            mean_color,
            -1,
        )

    return img_copy
