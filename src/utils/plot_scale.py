import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from typing import List, Tuple


def vertical_binning(
    boxes: List[List[int]], texts: List[str], width: int, bins: int = 10
) -> List[Tuple[List[int], str]]:
    """
    `boxes`: List[List[int]]
        `List` of bbox coordinates
    `texts`: List[str]
        `List` of OCR text detection
    `width`: int
        Image width
    `bins`: int
        Number of bins

    Returns
    `axes`: List[Tuple[List[int], str]]
        Predictions in bin with most predictions \
        (corresponds to vertical axes bin)
    `loc`: int
        Rightmost location of detections
    """
    win_size = int(width / bins)

    detects_bins = {}

    for i in range(bins):
        left = max(0, i * win_size - 20)
        right = (i + 1) * win_size

        detects = [
            (box, text) for box, text in zip(boxes, texts) if left <= box[0][0] <= right
        ]

        detects_bins[i] = {"left": left, "right": right, "detects": list(detects)}

    max_i = 0
    max_ = 0

    for k, v in detects_bins.items():
        if len(v["detects"]) > max_:
            max_ = len(v["detects"])
            max_i = k

    axes = dict(detects_bins[max_i])
    axes = axes["detects"]
    loc = max(ax[0][2][0] for ax in axes)
    top = min(ax[0][0][1] for ax in axes)

    return axes, loc, top


def preprocess_axes(axes: List[Tuple[List[int], str]]) -> pd.DataFrame:
    """
    `axes`: List[Tuple[List[int], str]]
        Axes bin

    Returns
    `axes_df`: pd.DataFrame
        Dataframe with coordinates and labels of axes tick labels
    """
    axes = [(box, int(text)) for box, text in axes if text.isnumeric()]
    axes_df = pd.DataFrame(axes)
    axes_df.columns = ["box", "depth"]
    axes_df["height"] = axes_df["box"].apply(lambda x: x[0][1])

    return axes_df


def get_outliers(axes: pd.DataFrame) -> np.ndarray:
    dbscan = DBSCAN(eps=0.05)

    axes = MinMaxScaler().fit_transform(axes[["depth", "height"]])

    classes = dbscan.fit_predict(axes)

    return classes


def get_trend(axes: pd.DataFrame) -> Tuple[float, float]:
    lr = Ridge()

    idx = axes["class"].value_counts().idxmax()

    sample = axes[axes["class"] == idx]

    lr.fit(sample[["height"]].to_numpy(), sample[["depth"]].to_numpy())

    return lr.coef_[0][0], lr.intercept_[0]


def compute_depth_scale(
    ocr_pred: Tuple[List[List[int]], List[str]],
    width: int,
):
    """
    `ocr_pred`: Tuple[List[List[int]], List[str]]
        Tuple of `List` of bboxes and `List` of text from OCR
    `width`: int
        Image width

    Returns
    `linreg`: Tuple[float, float]
        `Slope` & `Intercept` coefficient for LinReg
    """
    boxes, texts = ocr_pred
    bins = 10

    if width > 2000:
        bins = int(width / 200)

    axes, loc = vertical_binning(boxes, texts, width, bins)
    axes = preprocess_axes(axes)

    axes["class"] = get_outliers(axes)

    slope, intercept = get_trend(axes)

    return slope, intercept, loc
