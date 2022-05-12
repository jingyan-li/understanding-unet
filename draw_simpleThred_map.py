import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import random as rng
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

    # to store multiple windows layer
    layers = []
