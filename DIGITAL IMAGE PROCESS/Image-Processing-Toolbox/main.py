import matplotlib.pyplot as plt
import os
from src.filters import apply_median_filter, apply_gaussian_filter
from src.edge_detection import sobel_edge_detection
from src.image_enhancement import histogram_matching
from src.inpainting import process_inpainting
from PIL import Image
import numpy as np

# 📂 File Paths
SAMPLES_DIR = "samples/"
RESULTS_DIR = "results/"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Image Helper Function
def load_image(filename):
    """Loads an image and converts it to grayscale numpy array."""
    img = Image.open(filename).convert('L')
    return np.array(img, dtype=np.float32)

# Save Image Helper Function
def save_image(image, filename):
    """Saves a numpy image array as a file."""
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(filename)

# Display Image Function
def display_images(image_list, titles):
    """Displays multiple images side by side."""
    fig, axes = plt.subplots(1, len(image_list), figsize=(15, 5))

    for ax, img, title in zip(axes, image_list, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# ✅ 1. Apply Filtering (Gaussian & Median)
def test_filters(original_image):
    median_result = apply_median_filter(original_image, size=3)
    gaussian_result = apply_gaussian_filter(original_image, sigma=1.5)

    save_image(median_result, f"{RESULTS_DIR}/median_filter.png")
    save_image(gaussian_result, f"{RESULTS_DIR}/gaussian_filter.png")

    display_images([original_image, median_result, gaussian_result],
                   ["Original Image", "Median Filter", "Gaussian Filter"])

# ✅ 2. Apply Edge Detection (Sobel)
def test_edge_detection(original_image):
    Gx, Gy, gradient_magnitude = sobel_edge_detection(original_image)

    save_image(Gx, f"{RESULTS_DIR}/sobel_x.png")
    save_image(Gy, f"{RESULTS_DIR}/sobel_y.png")
    save_image(gradient_magnitude, f"{RESULTS_DIR}/gradient_magnitude.png")

    display_images([original_image, Gx, Gy, gradient_magnitude],
                   ["Original Image", "Sobel X", "Sobel Y", "Gradient Magnitude"])

# ✅ 3. Apply Image Inpainting
def test_inpainting(damaged_image_path, mask_image_path):
    output_path = f"{RESULTS_DIR}/restored_output.png"
    restored_image = process_inpainting(damaged_image_path, mask_image_path, output_path)

    display_images([load_image(damaged_image_path), restored_image],
                   ["Damaged Image", "Restored Image"])

# ✅ 4. Apply Histogram Matching
def test_histogram_matching(source_image, reference_image):
    matched_result = histogram_matching(source_image, reference_image)

    save_image(matched_result, f"{RESULTS_DIR}/histogram_matched.png")

    display_images([source_image, reference_image, matched_result],
                   ["Source Image", "Reference Image", "Histogram Matched"])

# 🚀 Run All Tests
if __name__ == "__main__":
    # Load sample images
    original_image = load_image(f"{SAMPLES_DIR}/sample1.jpg")
    damaged_image_path = f"{SAMPLES_DIR}/damaged_sample.jpg"
    mask_image_path = f"{SAMPLES_DIR}/mask_sample.jpg"
    reference_image = load_image(f"{SAMPLES_DIR}/reference.jpg")

    print("Running Filtering Tests...")
    test_filters(original_image)

    print("Running Edge Detection Tests...")
    test_edge_detection(original_image)

    print("Running Inpainting Tests...")
    test_inpainting(damaged_image_path, mask_image_path)

    print("Running Histogram Matching Tests...")
    test_histogram_matching(original_image, reference_image)

    print("\n✅ All tests completed! Check the results folder for saved images.")
