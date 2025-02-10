import numpy as np
from skimage.filters import median
from skimage.morphology import disk  # Using disk instead of rectangle

def apply_gaussian_filter(image, sigma=1):
    """Apply Gaussian smoothing filter."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=sigma)

def apply_median_filter(image, size=3):
    """Apply Median filter for noise removal."""
    return median(image, disk(size))  # Corrected: using disk instead of footprint_rectangle
