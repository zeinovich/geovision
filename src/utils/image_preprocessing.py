from PIL import Image
import cv2
from pdf2image import convert_from_bytes
import numpy as np
from paddleocr import PaddleOCR
import math
from typing import List, Tuple

import os
from aspose.imaging import Image as aspose_image
from aspose.imaging.imageoptions import *

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd



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
        return -1

    return image

def image_processing(
    img: np.ndarray, filter_kernel_size: int = 3, blur_kernel_size: int = 10, iterations: int = 1
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
        (filter_kernel_size, filter_kernel_size),
        dtype=np.uint8,
    )

    img = cv2.blur(img, (blur_kernel_size, blur_kernel_size))

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

def display_boxes(img: np.ndarray, boxes: List[List[int]]) -> np.ndarray:
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
    img_copy = np.array(img_copy)
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
            [255, 0, 0],
            2,
        )

    return img_copy





def pivot_data_for_visualization(
        df_strat: pd.DataFrame,
        col_reference: str,
        col_depth: str = 'DEPTH',
        depth_step: float = 0.5
) -> pd.DataFrame:
    """
    function to prepare the stratigraphy data for visualization
    Args:
        df_strat ():
        col_reference ():
        col_depth ():
        depth_step ():

    Returns:

    """

    # pivot table
    df_pivot = df_strat[[col_depth, col_reference]].dropna().pivot(index=col_depth,
                                                                   columns=col_reference,
                                                                   values=col_depth)
    # fillna with 0
    df_pivot = df_pivot.fillna(0)

    # replace all non na values with 100
    df_pivot[df_pivot != 0] = 100

    # reset index
    df_pivot = df_pivot.reset_index()

    # rename Кровля to MD
    df_pivot.rename(columns={col_depth: 'DEPTH'}, inplace=True)

    # resample the dataframe
    start, end = df_pivot['DEPTH'].min(), df_pivot['DEPTH'].max()
    new_md = list(np.arange(start, end + depth_step, depth_step))
    resampled_df = pd.DataFrame(new_md, columns=['DEPTH'])

    resampled_df = resampled_df.merge(df_pivot, on='DEPTH', how='outer').sort_values(
        by='DEPTH').ffill()

    return resampled_df

def logview(
        df_log:pd.DataFrame,
        df_lith_mixed:pd.DataFrame = pd.DataFrame(),
        df_lith_dominant:pd.DataFrame = pd.DataFrame(),
        df_formation:pd.DataFrame = pd.DataFrame(),
        features_to_log:list = [],
        col_depth: str = "DEPTH"
):
    """
    function to construct layout for the well
    Args:
        df_log ():
        df_lith_mixed ():
        df_lith_dominant ():
        df_formation ():
        features_to_log ():
        col_depth ():

    Returns:

    """
    # Base number of columns is the number of features in df_log
    num_cols = len(features_to_log) if features_to_log else df_log.shape[1]

    # Increment num_cols if additional dataframes are present
    if not df_lith_mixed.empty:
        num_cols += 1
    if not df_lith_dominant.empty:
        num_cols += 1
    if not df_formation.empty:
        num_cols += 1

    fig = make_subplots(rows=1, cols=num_cols, shared_yaxes=True)

    # specify features to plot
    features = [col for col in df_log.columns if col not in [col_depth] + ['WELL']]

    #cur subplot pos
    col_numbers = 0

    #plotting features
    for ix, feat in enumerate(features):
        fig.add_trace(
            go.Scatter(
                x=df_log[feat],
                y=df_log[col_depth],
                mode="lines",
                line=dict(color="black", width=0.5),
                name=feat,
            ),
            row=1,
            col=ix + 1,
        )

        fig.update_xaxes(
            title=dict(text=feat),
            row=1,
            col=ix + 1,
            side="top",
            tickangle=-90,
        )
        if feat in features_to_log:
            fig.update_xaxes(col=ix + 1, type="log")

        col_numbers += 1

    #plot lithology dominant
    if not df_lith_dominant.empty:
        cols_lith = df_lith_dominant.loc[:, df_lith_dominant.columns.str.startswith('LITHO_')].columns
        for jx, lith in enumerate(cols_lith):
            df_lith_codes = get_lithology_mapper()

            fig.add_trace(
                go.Scatter(
                    x=df_lith_dominant[lith],
                    y=df_lith_dominant[col_depth],
                    fill="tozerox",
                    mode="lines",
                    name=lith,
                    line=dict(color="grey", width=0.01),
                    fillcolor=df_lith_codes[lith]["color"],
                    fillpattern=dict(
                        shape=df_lith_codes[lith]["hatch"],
                    ),
                ),
                row=1,
                col=col_numbers + 1,
            )

            fig.update_xaxes(
                title="LITHOLOGY",
                row=1,
                col=col_numbers + 1,
                side="top",
                tickangle=-90,
            )
        col_numbers += 1

    #plot lithology mixed
    if not df_lith_mixed.empty:

        #get lith columns
        cols_lith = df_lith_mixed.loc[:, df_lith_mixed.columns.str.startswith('LITHO_')].columns

        # cumulative sum by raw
        df_lith_mixed[cols_lith] = df_lith_mixed[cols_lith].cumsum(axis=1)

        #visualize traces
        for jx, lith in enumerate(cols_lith):

            #init dict of lithology codes
            df_lith_codes = get_lithology_mapper()

            #init fill mode
            fill_mode = 'tozerox' if jx == 0 else 'tonextx'
            fig.add_trace(
                go.Scatter(
                    x=df_lith_mixed[lith],
                    y=df_lith_mixed[col_depth],
                    fill=fill_mode,
                    mode="lines",
                    name=lith,
                    line=dict(color="black", width=0.01),
                    fillcolor=df_lith_codes[lith]["color"],
                    fillpattern=dict(shape=df_lith_codes[lith]["hatch"],
                                     fgcolor='black'
                                     ),
                ),
                row=1,
                col=col_numbers + 1,
            )

            fig.update_xaxes(
                title="LITHOLOGY",
                # titlefont=dict(size=10),
                row=1,
                col=col_numbers + 1,
                side="top",
                tickangle=-90,
            )
        col_numbers += 1

    #plot formation tops
    if not df_formation.empty:
        features_to_drop = ["DEPTH"]
        cols_zone = [col for col in df_formation.columns if col not in features_to_drop]

        for jx, zone in enumerate(cols_zone):

            fig.add_trace(
                go.Scatter(
                    x=df_formation[zone],
                    y=df_formation[col_depth],
                    fill="tozerox",
                    mode="lines",
                    name=zone,
                ),
                row=1,
                col=col_numbers + 1,
            )


        fig.update_xaxes(
            title="Fromation tops",
            row=1,
            col=col_numbers + 1,
            side="top",
            tickangle=-90,
            automargin=True
        )
        col_numbers += 1

    fig.update_yaxes(
        title_text="DEPTH",
        row=1,
        col=1,
        autorange="reversed",
        tickformat=".0f",
    )
    fig.update_layout(height=900, width=1200, showlegend=False)

    return fig

