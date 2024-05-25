import streamlit as st

from paddleocr import PaddleOCR
from PIL import Image

from utils.image_preprocessing import image_processing, get_ocr
from utils.plot_scale import compute_depth_scale


def main():
    OCR_MODEL = PaddleOCR(
        ocr_version="PP-OCRv4",
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
    )

    image_name = st.text_input("Enter path to file:")

    image = Image.open(image_name)

    st.image(image)

    preds = get_ocr(OCR_MODEL, image)


if __name__ == "__main__":
    main()
