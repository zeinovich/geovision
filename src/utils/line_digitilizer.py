import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
import logging

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s"
)

logger = logging.getLogger(name=f"src.utils.line_digitilizer")

def digitilize_single_line(
        cropped_img: np.array,
        bbox: Tuple[int, int, int, int],
        depthK: float, depthB: float,
        mnemonicK: float, mnemonicB: float,
        ad_dbscan_eps=0.5, ad_dbscan_min_samples=3):
    """
    Keyword arguments:

    * cropped_img - opencv (H, W, C) format image contains section with lineplot (already preprocessed)
    * bbox - opencv bounding box (xmin, ymin, xmax, ymax) where (xmin, ymin) = TopLeft; (xmax, ymax) = BottomRight
    * k, b - coefficient from regression which translates pixel value inside
    """
    
    # First axis - y, second axis - x
    xmin, ymin, xmax, ymax = bbox
    depth_per_pixel = depthK * np.arange(ymin, ymax) + depthB
    mnemonic_per_pixel = mnemonicK * np.arange(xmin, xmax) + mnemonicB
    # [X]: asssertion
    assert cropped_img.shape[0] == len(depth_per_pixel)

    # Detect anomaly pixels
    cluster_labels = []
    for i in tqdm(range(cropped_img.shape[0]), desc='Line detection'):
        scaler = MinMaxScaler()
        clustering = DBSCAN(eps=ad_dbscan_eps, min_samples=ad_dbscan_min_samples).fit(
            scaler.fit_transform(cropped_img[i])
        )
        cluster_labels.append(clustering.labels_)
    cluster_labels = np.array(cluster_labels)
    cluster_labels[cluster_labels != -1] = 0
    cluster_labels[cluster_labels == -1] = 1
    logger.info(f'Found: {np.count_nonzero(cluster_labels)} line pixels')

    x_values = []
    y_values = []
    for i in range(cluster_labels.shape[0]):
        ixs = np.argwhere(cluster_labels[i] == 1).flatten()
        if len(ixs) > 0:
            x_values.append(
                mnemonic_per_pixel[np.random.choice(ixs, 3)[0]]
            )
            y_values.append(
                depth_per_pixel[i]
            )
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    return cluster_labels, x_values, y_values
