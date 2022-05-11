import random as rng
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from utils.IO import log_scale


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