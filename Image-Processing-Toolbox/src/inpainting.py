import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import median
from skimage.morphology import disk
from PIL import Image
from skimage.transform import resize

def load_image(filename, grayscale=True):
    """Loads an image and converts it to grayscale if needed."""
    img = Image.open(filename)
    if grayscale:
        img = img.convert("L")
    return np.array(img, dtype=np.float32)

def save_image(image, filename):
    """Saves an image after normalizing pixel values to [0,255]."""
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(filename)

def inpaint_image(damaged, mask, iterations=150, sigma=2.0):
    """
    Restores a damaged image using iterative Gaussian smoothing.
    Also applies median filtering to reduce visible artifacts.
    """
    inpainted = damaged.copy()

    for _ in range(iterations):
        blurred = gaussian_filter(inpainted, sigma=sigma, mode='constant')
        inpainted[mask == 0] = blurred[mask == 0]  # Replace only damaged pixels

    # Post-processing to reduce sharp edges:
    inpainted = median(inpainted, disk(3))  # Median filter to remove harsh edges
    inpainted = gaussian_filter(inpainted, sigma=1.5)  # Final blur for smoothness

    return inpainted

# ✅ This function will be called from `main.py`
def process_inpainting(damaged_path, mask_path, output_restored):
    """Loads images, performs inpainting, and saves the restored image."""
    damaged = load_image(damaged_path)
    mask = load_image(mask_path)

    # Ensure mask and damaged image have the same dimensions
    if mask.shape != damaged.shape:
        mask = resize(mask, damaged.shape, mode="reflect", anti_aliasing=True)

    # Convert mask to binary format (0 = damaged, 1 = undamaged)
    mask = (mask > 100).astype(np.uint8)  # Threshold lowered for smoother blending

    # Perform inpainting
    restored = inpaint_image(damaged, mask)

    # Save and return results
    save_image(restored, output_restored)
    return restored
