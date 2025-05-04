import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte

def compute_histogram_custom(image_path, n_bins=128):
    """
    Compute a custom histogram of an image without using built-in histogram functions.
    """
    image = io.imread(image_path, as_gray=True)
    image = img_as_ubyte(image)

    # Compute histogram manually
    hist = np.zeros(n_bins, dtype=int)
    bin_width = 256 / n_bins

    for pixel in image.ravel():
        bin_index = int(pixel // bin_width)
        hist[bin_index] += 1

    return hist