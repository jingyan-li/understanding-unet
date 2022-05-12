import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.IO import read_filtering_result
from utils.draw import generate_random_plot_colors


def main():
    # path
    figure_log_root = Path('/cluster/scratch/jingyli/interpretable/data_intermediate/visualization')

    MODEL = "resUnet"
    DATE = "2019-07-01"
    TIME = 144
    CHANNEL = 0
    method = "watershed"
    render_attribution = False
    # To store multiple layers
    layers = []


    idx = 1
    for i in tqdm(range(1,22,2)):
        for j in range(0,22,2):
            W = f"{i}_{j}"
            arr = read_filtering_result(DATE, TIME, CHANNEL, W, MODEL, figure_log_root=figure_log_root, method=method)
            if arr is None:
                continue
            if not render_attribution:
                markers = arr[...,0]
                markers = np.where((markers > 0) & (markers < 255), idx, 0)
                layers.append(markers)
            else:
                attr_values = arr[...,1]
                layers.append(attr_values)
            idx += 1
    layers = np.array(layers)
    print(f"Loaded {len(layers)} in total")


    colors = generate_random_plot_colors(layers.shape[0])

    # Create the result image
    dst = np.ones((layers.shape[1], layers.shape[2], 4), dtype=np.uint8) * 255
    # dst[..., 3] = np.ones(watershed_layer.shape[1:])*100
    # Fill labeled objects with random colors
    for i in range(layers.shape[1]):
        for j in range(layers.shape[2]):
            # Filter layers where has marker
            layers_with_marker = layers[:, i, j]
            markers = np.unique(layers_with_marker[np.nonzero(layers_with_marker)])
            if len(markers)==0:
                continue
            index = markers[-1]
            dst[i, j, :3] = colors[index - 1]

    # Visualize the final image
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    im = ax.imshow(dst)
    ax.set_title(f"{method} global map")
    plt.savefig(f"{method}.png",bbox_inches="tight")


if __name__ == "__main__":
    main()



