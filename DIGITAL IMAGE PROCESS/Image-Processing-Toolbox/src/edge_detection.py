import numpy as np
from scipy.signal import convolve2d

def sobel_edge_detection(image):
    """Applies Sobel filters for edge detection."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
    Gy = convolve2d(image, sobel_y, mode='same', boundary='symm')

    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)
    
    return Gx, Gy, gradient_magnitude
