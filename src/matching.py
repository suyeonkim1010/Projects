from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from src.histogram_utils import compute_histogram_custom

def match_image_histogram(source_path, reference_path, n_bins=128):
    """
    Match the histogram of the source image to a reference image.
    """
    source = io.imread(source_path, as_gray=True)
    source = img_as_ubyte(source)
    reference = io.imread(reference_path, as_gray=True)
    reference = img_as_ubyte(reference)

    hist_src = compute_histogram_custom(source_path, n_bins)
    hist_ref = compute_histogram_custom(reference_path, n_bins)

    cdf_src = np.cumsum(hist_src) / np.sum(hist_src)
    cdf_ref = np.cumsum(hist_ref) / np.sum(hist_ref)

    # Compute mapping
    mapping = np.zeros(n_bins, dtype=np.uint8)
    ref_idx = 0
    for src_idx in range(n_bins):
        while ref_idx < n_bins - 1 and cdf_ref[ref_idx] < cdf_src[src_idx]:
            ref_idx += 1
        mapping[src_idx] = ref_idx

    # Apply mapping
    matched_image = mapping[(source / 2).astype(np.uint8)]

    return matched_image
