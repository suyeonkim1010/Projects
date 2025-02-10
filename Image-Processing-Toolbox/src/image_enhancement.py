import numpy as np
from skimage.exposure import match_histograms

def histogram_matching(source, reference):
    """Match the histogram of the source image to the reference image."""
    if source.ndim == 3 and reference.ndim == 3:
        matched = np.stack([
            match_histograms(source[:, :, i], reference[:, :, i])  # Match each channel separately
            for i in range(source.shape[-1])
        ], axis=-1)
    else:
        matched = match_histograms(source, reference)  # Single-channel case
    return matched
