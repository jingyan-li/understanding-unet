from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm

from utils.IO import read_attr, get_space_mat, normalize_space_mat
from utils.draw import plot_attr_map


def main(MODEL, DATE, TIME, CHANNEL, W,
         attribution_root, figure_log_root,
         channel_of_interest: int = 11, quantile: float = 0.99,
         visualize: bool = True,
         save_visualize: bool = True,
         save_pickle: bool = True
         ):
    ## Read Attribution
    # path
    attribution_root = Path(attribution_root)
    figure_log_root = Path(figure_log_root)

    attr, figure_log_path = read_attr(DATE, TIME, CHANNEL, W, MODEL, attribution_root, figure_log_root, method_dir="simpleThreshold")

    # print(figure_log_path)

    if attr is None:
        # print("Not exists")
        return

    filename = f"simpleThreshold_{DATE}_{TIME}_C{CHANNEL}-W{W}" if channel_of_interest == 11 else f"simpleThreshold_{DATE}_{TIME}_C{CHANNEL}-W{W}-feat{channel_of_interest}"

    space_mat = get_space_mat(attr)
    # of shape [19,512,448]
    # 19 = 12 timeepochs + 7 static features

    # Normalize space mat such that in each channel, the attribution lies in the range [0,255]
    norm_space_mat = normalize_space_mat(space_mat)  # of shape [512,448,19]

    # Find the threshold value defined by the quantile
    flat_channel = norm_space_mat[..., channel_of_interest].flatten()
    flat_channel = np.sort(flat_channel)
    threshold = flat_channel[np.round(quantile * flat_channel.shape[0]).astype(int)]
    # print(f"Threshold {threshold}")

    if visualize:
        import matplotlib as mpl

        # Mask out values below threshold
        channel = norm_space_mat[..., channel_of_interest]
        channel = np.ma.masked_where(channel < threshold, channel)

        cmap = mpl.cm.get_cmap("OrRd").copy()
        cmap.set_bad(color="white")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_attr_map(channel, ax, cmap)

        # Save plot
        if save_visualize != None:
            plt.savefig(figure_log_path / f"{filename}.png", bbox_inches='tight')
            plt.close(fig)
        del fig, ax

    if save_pickle != None:
        import pickle
        channel = norm_space_mat[..., channel_of_interest]
        channel[channel < threshold] = -99
        np.save(figure_log_path / filename, channel)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description='conduct simple threshold')
    a.add_argument('--attribution_root',
                   default='/cluster/scratch/jingyli/interpretable/attribution_result/', type=str,
                   help='root path to attribution pickle')
    a.add_argument('--figure_log_root',
                   default='/cluster/scratch/jingyli/interpretable/data_intermediate/visualization', type=str,
                   help='root path to save figures')
    a.add_argument('--quantile', default=0.995, type=float,
                   help='quantile as threshold to filter out hotspots')
    a.add_argument('--channel_of_interest', default=11, type=int,
                   help='which feature to be conducted watershed [0,18]')
    a.add_argument('--save_pickle', default=True, type=bool,
                   help='whether to save watershed and attr map as .npy')
    a.add_argument('--visualize', default=True, type=bool,
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

    # main(MODEL, DATE, TIME, CHANNEL, W="16_18", **vars(args))