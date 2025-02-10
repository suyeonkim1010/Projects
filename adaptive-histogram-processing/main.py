from src.equalization import enhance_image_contrast
from src.matching import match_image_histogram
from src.visualization import plot_histogram
from skimage import io
import matplotlib.pyplot as plt

# File paths
source_image = "images/sunset.jpg"
reference_image = "images/city_night.jpg"

# Plot the histogram of the source image
plot_histogram(source_image)

# Apply histogram equalization to the source image
equalized_img = enhance_image_contrast(source_image)
plt.figure(figsize=(8, 4))
plt.imshow(equalized_img, cmap='gray')
plt.title("Histogram Equalized Image")
plt.show()

# Perform histogram matching between source and reference images
matched_img = match_image_histogram(source_image, reference_image)
plt.figure(figsize=(8, 4))
plt.imshow(matched_img, cmap='gray')
plt.title("Histogram Matched Image")
plt.show()

