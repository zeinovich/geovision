import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image

import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os

from utils import image_preprocessing


def main():

    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.set_page_config(layout='wide')
    st.sidebar.markdown('# GeoVision')

    with st.sidebar:
        selected_step = option_menu(
            menu_title='',
            options=[
                'Step 1: Upload image',
                'Step 2: Image processing',
                'Step 3: Result'
            ],
            menu_icon='display',
            default_index=0
        )

    uploaded_file = st.sidebar.file_uploader("Upload image", type=['jpg', 'jpeg', 'png', 'pdf', 'tiff', 'cdr'])

    if selected_step == 'Step 1: Upload image':
        if uploaded_file:

            st.title("Uploaded file")
            # image = Image.open(uploaded_image)

            image = image_preprocessing.load_and_display_file(uploaded_file)
            st.image(image, use_column_width=True)

    if selected_step == 'Step 2: Image processing':
        pass

    if selected_step == 'Step 3: Result':
        pass


if __name__ == "__main__":
    main()