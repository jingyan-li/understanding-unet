import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import random as rng
import argparse
from tqdm import tqdm


# functions to read attribution and format attribution matrix
def read_attr(DATE, TIME, CHANNEL, W, MODEL, attribution_root:Path, figure_log_root:Path):
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
        Normalize each dim into [0,255] (unit8)

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


def plot_attr_map(attr_dim, ax):
    """
        Plot attribution map of a dimension
    """
    im = ax.imshow(attr_dim, cmap='RdBu_r', vmin=0, vmax=255)
    ax.set_title("Attribution map")


def plot_attribution_hist(attr_arr):
    """
        Plot histogram of a list of attribution levels
    """
    plt.hist(attr_arr.ravel(), bins=100)
    plt.yscale('log')
    plt.show()


def log_scale(arr):
    """
        Make the 2D array to log scale; Normalize it into [0,255]
    """
    arr_ = np.log1p(arr)
    # Normalize
    arr_ = (arr_ - np.min(arr_)) / (np.max(arr_) - np.min(arr_)) * 255
    arr_ = np.round(arr_, 0).astype("uint8")
    return arr_


def watershed(t, binary_thred=5, dist_peak_thred=0.5, visualize_inter=False, visualize_final=True):
    """
        Run watershed algorithm over t
        @binary_thred: int [0,255]
        @dist_peak_thred: float [0.,1.]
        return: markers learned after watershed
    """
    # Create binary image from source image
    _, bw = cv.threshold(t.astype("uint8"), binary_thred, 255, cv.THRESH_BINARY)
    bw = bw.astype("uint8")

    # Perform the distance transform algorithm
    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    # Visualize distance transform image
    if visualize_inter:
        plt.imshow(dist, cmap='gray')

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv.threshold(dist, dist_peak_thred, 1.0, cv.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    # Visualize peaks in distance tranform
    if visualize_inter:
        plt.imshow(dist, cmap='gray')

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')
    # Find total markers
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    markers_8u = (markers * 10).astype('uint8')

    # Perform the watershed algorithm
    cv.watershed(np.repeat(t, repeats=3, axis=2), markers)

    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)

    if visualize_final:
        # Generate random colors
        colors = []
        for contour in contours:
            colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
        # Create the result image
        dst = np.ones((markers.shape[0], markers.shape[1], 3), dtype=np.uint8) * 255
        # Fill labeled objects with random colors
        for i in range(markers.shape[0]):
            for j in range(markers.shape[1]):
                index = markers[i, j]
                if index > 0 and index <= len(contours):
                    dst[i, j, :] = colors[index - 1]
        # Visualize the final image
        plt.imshow(dst)
        plt.title(f"Watershed segmentations: {len(contours)}")

    return markers, contours


def plot_attr_over_watershed(attr, markers, contours, ax):
    """
        Plot attribution map over watershed segmentation
    """
    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    # Create the result image
    dst = np.ones((markers.shape[0], markers.shape[1], 4), dtype=np.uint8) * 255
    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :3] = colors[index - 1]
    # Alpha denotes the attribution level
    dst[..., -1] = log_scale(np.squeeze(attr, axis=-1))
    ax.imshow(dst)
    ax.set_title(f"Watershed segmentations: {len(contours)}")


def main(MODEL, DATE, TIME, CHANNEL, W,
         attribution_root, figure_log_root, binary_thred, dist_peak_thred,
            feat_index, save_pickle, attr_to_log
         ):
    attr, figure_log_path = read_attr(DATE, TIME, CHANNEL, W, MODEL, attribution_root, figure_log_root)
    if attr is None:
        return

    filename = f"{DATE}_{TIME}_C{CHANNEL}-W{W}" if feat_index == 11 else f"{DATE}_{TIME}_C{CHANNEL}-W{W}-feat{feat_index}"

    space_mat = get_space_mat(attr)
    # of shape [19,512,448]
    # 19 = 12 timeepochs + 7 static features

    space_mat_ = normalize_space_mat(space_mat, log_norm=attr_to_log)

    # Dimension for watershed
    n = args.feat_index
    select_dim_attr = space_mat_[:, :, n:n + 1]

    # ##### Watershed #######

    # Parameters
    BINARY_THRED = binary_thred
    DIST_PEAK_THRED = dist_peak_thred

    mark, contours = watershed(select_dim_attr, binary_thred=BINARY_THRED, dist_peak_thred=DIST_PEAK_THRED,
                               visualize_final=False)
    # print(f"WATERSHED: detected {len(contours)}")

    if save_pickle:
        np.save(figure_log_path / filename,
                np.concatenate((mark[..., np.newaxis], select_dim_attr), axis=-1))

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    plot_attr_map(select_dim_attr, axes[0])
    plot_attr_over_watershed(select_dim_attr, mark, contours, axes[1])
    plt.savefig(figure_log_path / f"{filename}.png", bbox_inches='tight')
    plt.close(fig)
    del fig, axes


if __name__ == "__main__":
    rng.seed(12345)

    a = argparse.ArgumentParser(description='conduct watershed')
    a.add_argument('--attribution_root',
                   default='/Users/aaaje/Documents/ETH_WORK/interpretability/data_intermediate/attribution', type=str,
                   help='root path to attribution pickle')
    a.add_argument('--figure_log_root',
                   default='/Users/aaaje/Documents/ETH_WORK/interpretability/data_intermediate/visualization', type=str,
                   help='root path to save figures')
    a.add_argument('--binary_thred', default=10, type=int,
                   help='threshold to binarize attribution map in watershed [0,255]')
    a.add_argument('--dist_peak_thred', default=0.5, type=float,
                   help='threshold to filter distance peak as markers in watershed [0,1]')
    a.add_argument('--feat_index', default=11, type=int,
                   help='which feature to be conducted watershed [0,18]')
    a.add_argument('--save_pickle', default=True, type=bool,
                   help='whether to save watershed and attr map as .npy')
    a.add_argument('--attr_to_log', default=True, type=bool,
                   help='whether to take log of attribution level')
    args = a.parse_args()

    # path
    attribution_root = Path(args.attribution_root)
    figure_log_root = Path(args.figure_log_root)

    print(args)

    MODEL = "resUnet"
    DATE = "2019-07-01"
    TIME = 144
    CHANNEL = 0
    
    for i in tqdm(range(22)):
        for j in range(22):
            W = f"{i}_{j}"
            main(MODEL, DATE, TIME, CHANNEL, W, **args)
            
