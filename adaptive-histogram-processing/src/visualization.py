import matplotlib.pyplot as plt
import numpy as np
from src.histogram_utils import compute_histogram_custom

def plot_histogram(image_path, n_bins=128):
    """
    Plot the histogram of a given image.
    """
    hist = compute_histogram_custom(image_path, n_bins)
    plt.figure(figsize=(8, 4))
    plt.bar(range(n_bins), hist, width=1, color='gray')
    plt.title(f'Histogram ({n_bins} bins)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
