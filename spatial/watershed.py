import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random as rng
import argparse
from tqdm import tqdm

from utils.IO import read_attr, get_space_mat, normalize_space_mat, error_message
from utils.draw import plot_attr_map
from utils.modules.watershed import watershed, plot_attr_over_watershed


def main(MODEL, DATE, TIME, CHANNEL, W,
         attribution_root, figure_log_root, binary_thred, dist_peak_thred,
         channel_of_interest, save_pickle, attr_to_log
         ):
    # path
    attribution_root = Path(attribution_root)
    figure_log_root = Path(figure_log_root)

    attr, figure_log_path = read_attr(DATE, TIME, CHANNEL, W, MODEL, attribution_root, figure_log_root, method_dir="watershed")
    if attr is None:
        print(error_message("Not exists", f"{figure_log_path}"))
        return

    filename = f"watershed_{DATE}_{TIME}_C{CHANNEL}-W{W}" if channel_of_interest == 11 else f"watershed_{DATE}_{TIME}_C{CHANNEL}-W{W}-feat{channel_of_interest}"

    space_mat = get_space_mat(attr)
    # of shape [19,512,448]
    # 19 = 12 timeepochs + 7 static features

    space_mat_ = normalize_space_mat(space_mat, log_norm=attr_to_log)

    # Dimension for watershed
    n = channel_of_interest
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
                   default='/cluster/scratch/jingyli/interpretable/attribution_result/', type=str,
                   help='root path to attribution pickle')
    a.add_argument('--figure_log_root',
                   default='/cluster/scratch/jingyli/interpretable/data_intermediate/visualization', type=str,
                   help='root path to save figures')
    a.add_argument('--binary_thred', default=20, type=int,
                   help='threshold to binarize attribution map in watershed [0,255]')
    a.add_argument('--dist_peak_thred', default=0.25, type=float,
                   help='threshold to filter distance peak as markers in watershed [0,1]')
    a.add_argument('--channel_of_interest', default=11, type=int,
                   help='which feature to be conducted watershed [0,18]')
    a.add_argument('--save_pickle', default=True, type=bool,
                   help='whether to save watershed and attr map as .npy')
    a.add_argument('--attr_to_log', default=True, type=bool,
                   help='whether to take log of attribution level')
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
            
