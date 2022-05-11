import matplotlib.pyplot as plt


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