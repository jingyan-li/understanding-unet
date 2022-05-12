import matplotlib.pyplot as plt
import random as rng

def plot_attr_map(attr_dim, ax, cmap="RdBu_r"):
    """
        Plot attribution map of a dimension
    """
    im = ax.imshow(attr_dim, cmap=cmap, vmin=0, vmax=255)
    ax.set_title("Attribution map")


def plot_attribution_hist(attr_arr):
    """
        Plot histogram of a list of attribution levels
    """
    plt.hist(attr_arr.ravel(), bins=100)
    plt.yscale('log')
    plt.show()


def generate_random_plot_colors(color_num: int = 5):
    rng.seed(12345)
    # Generate random colors
    colors = []
    for i in range(color_num):
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    return colors