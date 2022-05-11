import numpy as np
import os
from pathlib import Path

# functions to read attribution and format attribution matrix
def read_attr(DATE, TIME, CHANNEL, W, MODEL, attribution_root:Path, figure_log_root:Path):
    """
    READ Attrbution of shape [C, W, H]
    """
    file_path = f"{DATE}_berlin_9ch{TIME}-saliency-target-channel{CHANNEL}-W{W}.npy"
    log_root = attribution_root / f"{MODEL}" / f"{DATE}_{TIME}"
    if not os.path.exists(log_root / file_path):
        return None, None

    attr = np.load(log_root / file_path)[0]
    figure_log_path = figure_log_root / f"{MODEL}" / f"{DATE}_{TIME}" / "watershed"
    if not os.path.exists(figure_log_path):
        os.makedirs(figure_log_path)
    return attr, figure_log_path


def get_space_mat(attr):
    """
    Convert attribution tensor to [12*timeepoch+7*statics, H, W]
    """
    # Get contribution per timeepoch
    C, H, W = attr.shape
    time_attr = np.sum(attr[:108].reshape(12, -1, H, W)[:, :-1, ...].reshape(12, -1, H, W), axis=1)
    # Get contribution for static features
    static_attr = attr[108:]
    # Stack dynamic + static
    stacked_attr = np.concatenate((time_attr, static_attr), axis=0)

    return stacked_attr


def normalize_space_mat(space_mat, log_norm=False):
    """
        Normalize along channel dim into [0,255] (unit8)

    """
    if log_norm:
        space_mat = np.log1p(space_mat)
    mins = np.amin(space_mat, axis=(1, 2))[:, np.newaxis, np.newaxis]
    maxs = np.amax(space_mat, axis=(1, 2))[:, np.newaxis, np.newaxis]

    space_mat_ = (space_mat - mins) / (maxs - mins) * 255

    # Round to integer
    space_mat_ = np.round(space_mat_, 0).astype('uint8')

    # Change the axis sequence to  [H,W,C]
    space_mat_ = np.moveaxis(space_mat_, (0, 1, 2), (2, 0, 1))

    return space_mat_


def log_scale(arr):
    """
        Make the 2D array to log scale; Normalize it into [0,255]
    """
    arr_ = np.log1p(arr)
    # Normalize
    arr_ = (arr_ - np.min(arr_)) / (np.max(arr_) - np.min(arr_)) * 255
    arr_ = np.round(arr_, 0).astype("uint8")
    return arr_