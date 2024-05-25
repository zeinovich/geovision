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
from utils.line_digitilizer import digitilize_single_line

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
            slope, intercept, bboxes = compute_depth_scale(
                (boxes, text_in_boxes), image.shape[1], image.shape[0]
            )

            image_with_boxes = display_boxes(image, boxes)
            # image_with_axes = Image.fromarray(image_with_boxes)
            # st.image(
            #     image_with_axes,
            # )

            # st.image(image_with_boxes[(bboxes[0][1]) : bboxes[0][3], bboxes[0][0]:bboxes[0][2]])


            list_res = []
            for idx, bbox in enumerate(bboxes):
                img_crop = image_with_boxes[bbox[1] : bbox[3], bbox[0]:bbox[2]]
                st.title(f'Crop {idx+1}')
                st.image(img_crop)
                _, x_vals, y_vals = digitilize_single_line(
                    img_crop, bbox, depthK=slope, depthB=intercept
                )
                list_res.append((x_vals, y_vals))

            st.write(np.array(list_res))

            res_df = pd.DataFrame(
                {'DEPTH': y_vals*slope + intercept,}
            )
            for i, feature in enumerate(list_res):
                res_df[f'FEATURE{i}'] = feature[0]



            st.dataframe(res_df)

            fig = logview(res_df, col_depth="DEPTH")
            st.plotly_chart(fig, use_container_width=True)
            res_df.to_csv('data/res_crop.csv', index=False)



    if selected_step == "Step 3: Result":

        # st.write(list_res)
        #
        #
        # res_df = pd.DataFrame()

        # path_to_las = st.text_input("Enter path to las file")
        #
        # res_df = ls.read(path_to_las).df().reset_index()
        st.dataframe(res_df)

        # table = pivot_data_for_visualization(res_df, col_reference="FEATURE", depth_step=1)
        fig = logview(res_df, col_depth="DEPTH")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
