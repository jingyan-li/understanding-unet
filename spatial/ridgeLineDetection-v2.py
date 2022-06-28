# TODO: Opencv Ridge Detection test
import cv2 as cv
import skimage

from utils.conversion import digit_image_to_int

print(skimage.__version__)
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse
from tqdm import tqdm
from pathlib import Path
from utils.IO import read_attr, get_space_mat, normalize_space_mat


def detect_ridges(gray, sigma=3.0):
    H_elems = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(H_elems)
    return i1, i2


def draw_ax(img, ax, title):
    im = ax.imshow(img, cmap="Greys", vmin=0, vmax=255)
    ax.set_title(title)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def main(MODEL, DATE, TIME, CHANNEL, W,
         attribution_root, figure_log_root,
         channel_of_interest: int = 11, sigma: float = 3.0,
         visualize: bool = False,
         save_visualize: bool = True,
         save_pickle: bool = True):
    ## Read Attribution
    # path
    attribution_root = Path(attribution_root)
    figure_log_root = Path(figure_log_root)
    attr, figure_log_path = read_attr(DATE, TIME, CHANNEL, W, MODEL, attribution_root, figure_log_root,
                                      method_dir="ridge")
    # print(figure_log_path)
    if attr is None:
        # print("Not exists")
        return
    filename = f"ridge_{DATE}_{TIME}_C{CHANNEL}-W{W}" if channel_of_interest == 11 else f"ridge_{DATE}_{TIME}_C{CHANNEL}-W{W}-feat{channel_of_interest}"

    s = get_space_mat(attr)
    norm_s = normalize_space_mat(s, log_norm=True)[..., channel_of_interest]
    I = norm_s  # np.asarray(Image.open('../utils/test.tiff'))
    i1, i2 = detect_ridges(I, sigma=sigma)
    local_maxima = digit_image_to_int(i1.copy())
    local_minima = 255-digit_image_to_int(i2.copy())
    # local_minima[local_minima<0] = 0

    if visualize or save_visualize:
        fig, axes = plt.subplots(1,3, figsize=(24,8))
        draw_ax(local_maxima, axes[0], "Maxima Ridges")
        draw_ax(local_minima, axes[1], "Minima Ridges")
        draw_ax(I, axes[2], "Attribution Map")
        if visualize:
            plt.show()
        else:
            plt.savefig(figure_log_path / f"{filename}.png", bbox_inches='tight')
            plt.close(fig)
        del fig, axes

    if save_pickle:
        import pickle
        np.save(figure_log_path / filename, local_minima)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description='conduct ridgeline detection')
    a.add_argument('--attribution_root',
                   default='/cluster/scratch/jingyli/interpretable/attribution_result/', type=str,
                   help='root path to attribution pickle')
    a.add_argument('--figure_log_root',
                   default='/cluster/scratch/jingyli/interpretable/data_intermediate/visualization', type=str,
                   help='root path to save figures')
    a.add_argument('--sigma', default=3.0, type=float,
                   help='quantile as threshold to filter out hotspots')
    a.add_argument('--channel_of_interest', default=11, type=int,
                   help='which feature to be conducted watershed [0,18]')
    a.add_argument('--save_pickle', default=True, type=bool,
                   help='whether to save watershed and attr map as .npy')
    a.add_argument('--visualize', default=False, type=bool,
                   help='whether to visualize the attribution map')
    a.add_argument('--save_visualize', default=True, type=bool,
                   help='whether to save the visualization of the attribution map')
    args = a.parse_args()

    print(args)

    MODEL = "resUnet"
    DATE = "2019-07-01"
    TIME = 144
    CHANNEL = 0

    for i in tqdm(range(22)):
        for j in range(22):
            W = f"{i}_{j}"
            main(MODEL, DATE, TIME, CHANNEL, W, **vars(args))