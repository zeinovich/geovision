import pandas as pd
import streamlit as st
from paddleocr import PaddleOCR
from PIL import Image

from utils.image_preprocessing import (
    image_processing,
    get_ocr,
    cover_detections,
    display_boxes,
    display_axes,
    pivot_data_for_visualization,
    logview,
)
from utils.plot_scale import compute_depth_scale
from streamlit_option_menu import option_menu

from PIL import Image

import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os

from utils import image_preprocessing

import plotly.express as px
import lasio as ls


def main():

    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.set_page_config(layout="wide")
    st.sidebar.markdown("# GeoVision")

    with st.sidebar:
        selected_step = option_menu(
            menu_title="",
            options=[
                "Step 1: Upload image",
                "Step 2: Image processing",
                "Step 3: Result",
            ],
            menu_icon="display",
            default_index=0,
        )

    uploaded_file = st.sidebar.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "pdf", "tiff", "cdr"]
    )
    image = None
    if uploaded_file:
        image = image_preprocessing.load_and_display_file(uploaded_file)

    if selected_step == "Step 1: Upload image":
        if uploaded_file:
            st.title("Uploaded file")

            if image == -1:
                st.error("Неподдерживаемый формат файла")
            else:
                st.image(image, use_column_width=True)

    if selected_step == "Step 2: Image processing":
        if image:
            st.title("Image processing stages")
            OCR_MODEL = PaddleOCR(
                ocr_version="PP-OCRv4",
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False,
            )

            boxes, text_in_boxes = get_ocr(OCR_MODEL, image)

            image = np.array(image)

            # left is leftmost point of axis detections
            # for axes in the middle
            slope, intercept, right, left, top = compute_depth_scale(
                (boxes, text_in_boxes), image.shape[1]
            )

            st.write([slope, intercept, right, left, top])

            image_with_boxes = display_boxes(image, boxes)
            image_with_axes = display_axes(image_with_boxes, right, top)
            image_with_axes = Image.fromarray(image_with_axes)
            st.image(
                image_with_axes,
            )





    if selected_step == "Step 3: Result":
        # res_df = pd.DataFrame({
        #     'DEPTH': np.arange(10000),
        #     'FEATURE': np.random.random(10000),
        #     'FEATURE2' : np.random.random(10000)
        # })

        path_to_las = st.text_input('Enter path to las file')

        res_df = ls.read(path_to_las).df().reset_index()
        st.dataframe(res_df)

        # table = pivot_data_for_visualization(res_df, col_reference="FEATURE", depth_step=1)
        fig = logview(res_df, col_depth='DEPT')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
