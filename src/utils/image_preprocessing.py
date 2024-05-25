from PIL import Image
import cv2
from pdf2image import convert_from_bytes
import numpy as np
from paddleocr import PaddleOCR
import math
from typing import List, Tuple


def load_and_display_file(uploaded_file):
    file_type = uploaded_file.type.split('/')[-1].lower()

    if file_type in ['jpg', 'jpeg', 'png', 'tiff', 'tif']:
        image = Image.open(uploaded_file)
    elif file_type == 'pdf':
        # Используем pdf2image для конвертации PDF в изображение
        images = convert_from_bytes(uploaded_file.read())
        image = images[0]  # Берем первую страницу PDF
    elif file_type == 'cdr':
        return None
    else:
        st.error("Неподдерживаемый формат файла")
        return None

    return image

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
    img = np.array(img)
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
    img = np.array(img)
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

