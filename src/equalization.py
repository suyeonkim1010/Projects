from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from src.histogram_utils import compute_histogram_custom

def enhance_image_contrast(image_path, n_bins=128):
    """
    Perform histogram equalization to enhance image contrast.
    """
    image = io.imread(image_path, as_gray=True)
    image = img_as_ubyte(image)

    hist = compute_histogram_custom(image_path, n_bins)
    cdf = np.cumsum(hist) / np.sum(hist)

    # Apply histogram equalization
    equalized_img = np.zeros_like(image)
    bin_width = 256 / n_bins
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            bin_index = int(pixel_value // bin_width)
            equalized_img[i, j] = cdf[bin_index] * 255

    return equalized_img
