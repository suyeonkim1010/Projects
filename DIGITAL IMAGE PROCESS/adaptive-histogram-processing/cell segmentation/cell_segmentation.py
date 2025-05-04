# import statements
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_ubyte
from scipy.ndimage.filters import minimum_filter
import scipy.ndimage

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성

def detect_blob_centers():
    
    imfile = 'nuclei.png'
    I = io.imread(imfile)

    # Apply Gaussian filter
    sig = 2.5

    J = filters.gaussian(I, sig)

    # # Plotting Input image and Blurred image
    plt.figure(figsize=(10,8))
    plt.subplot(121),plt.imshow(I,cmap='jet'), plt.title('Input Image')
    plt.subplot(122),plt.imshow(J,cmap='jet'),plt.title('Blurred Image')
    plt.savefig(os.path.join(output_dir, "Input+Blurred.png"))
    plt.show()


    # =========== 1. Create DoG volume ===========
    sigmas = [1.6 ** i for i in range(1, 5)]
    h, w = J.shape
    DoG = np.zeros([h, w, 3])
    
    DoG[:, :, 0] = filters.gaussian(J, sigmas[0]) - filters.gaussian(J, sigmas[1])
    DoG[:, :, 1] = filters.gaussian(J, sigmas[1]) - filters.gaussian(J, sigmas[2])
    DoG[:, :, 2] = filters.gaussian(J, sigmas[2]) - filters.gaussian(J, sigmas[3])
    
    level1 = DoG[:, :, 0]
    level2 = DoG[:, :, 1]
    level3 = DoG[:, :, 2]
    
    # # Plot DoG Levels
    plt.figure(figsize=(10,8))
    plt.subplot(131), plt.imshow(level1, cmap='jet'), plt.title('Level 1')
    plt.subplot(132), plt.imshow(level2, cmap='jet'), plt.title('Level 2')
    plt.subplot(133), plt.imshow(level3, cmap='jet'), plt.title('Level 3')
    plt.savefig(os.path.join(output_dir, "Level123.png"))

    plt.show()


    # =========== 2. Obtain a rough estimate of blob center locations ===========
    scatter_size = 40
    scatter_col = 'r'
    
    local_minima = np.zeros_like(DoG)

    local_minima[..., 0] = scipy.ndimage.minimum_filter(DoG[:, :, 0], size=(22,22))
    local_minima[..., 1] = scipy.ndimage.minimum_filter(DoG[:, :, 1], size=(24,24))
    local_minima[..., 2] = scipy.ndimage.minimum_filter(DoG[:, :, 2], size=(25,25)) 


    # Plotting local minima maps
    plt.figure(figsize=(10,8))
    plt.subplot(131), plt.imshow(local_minima[..., 0], cmap='jet')
    plt.subplot(132), plt.imshow(local_minima[..., 1], cmap='jet')
    plt.subplot(133), plt.imshow(local_minima[..., 2], cmap='jet')
    plt.savefig(os.path.join(output_dir, "local_minima.png"))

    plt.show()

    A = (DoG == local_minima).astype(int)

    B = np.sum(A, axis=2)
    
    y, x = np.nonzero(B)
    
    plt.figure(figsize=(10,8))
    plt.imshow(I, cmap='jet')
    plt.scatter(x, y, marker='.', color=scatter_col, s=scatter_size)
    plt.xlim([0, I.shape[1]])
    plt.ylim([0, I.shape[0]])
    plt.title('Rough Blob Centers Detected in Image')
    plt.savefig(os.path.join(output_dir, "Rough_Blob_Centers_Detected.png"))

    plt.show()

    # =========== 3. Refine the blob centers using Li thresholding ===========

    # Apply Gaussian filtering using skimage.filters.gaussian with a suitably chosen sigma and convert to unit8
    J = img_as_ubyte(J)

    threshold = filters.threshold_li(J)
    
    final = np.copy(B)
    final[J < threshold] = 0
    
    y, x = np.nonzero(final)


    # # Plotting
    plt.figure(figsize=(10,8))
    plt.imshow(I,cmap='jet')
    plt.scatter(x, y, marker='.', color=scatter_col, s=scatter_size)
    plt.xlim([0, I.shape[1]])
    plt.ylim([0, I.shape[0]])
    plt.title('Refined blob centers detected in the image')
    plt.savefig(os.path.join(output_dir, "Refined_blob_centers_detected.png"))

    plt.show()


    return final

# ----------------------------- PART 2 -----------------------------

def getSmallestNeighborIndex(img, row, col):
    """
    Parameters : 
    img            - image
    row            - row index of pixel
    col            - col index of pixel
    
    Returns         :  The location of the smallest 4-connected neighbour of pixel at location [row,col]

    """

    min_row_id = -1
    min_col_id = -1
    min_val = np.inf
    h, w = img.shape
    for row_id in range(row - 1, row + 2):
        if row_id < 0 or row_id >= h:
            continue
        for col_id in range(col - 1, col + 2):
            if col_id < 0 or col_id >= w:
                continue
            if row_id == row and col_id == col:
                continue
            if is_4connected(row, col, row_id, col_id):
              if img[row_id, col_id] < min_val:
                  min_row_id = row_id
                  min_col_id = col_id
                  min_val = img[row_id, col_id]     
    return min_row_id, min_col_id


def is_4connected(row, col, row_id, col_id):

    """
    Parameters : 
    row           - row index of pixel
    col           - col index of pixel
    row_id        - row index of neighbour pixel 
    col_id        - col index of neighbour pixel 
    
    Return         :  Boolean. Whether pixel at location [row_id, col_id] is a 4 connected neighbour of pixel at location [row, col]

    """

    return abs(row - row_id) + abs(col - col_id) == 1


def getRegionalMinima(img):
    """
    Parameters:
    img - input image

    Returns:
    markers - 32-bit int image with non-zero labels for local minima and zero elsewhere
    """
    h, w = img.shape
    markers = np.zeros((h, w), dtype=np.int32)
    current_label = 1  # Start labeling from 1

    for row in range(h):
        for col in range(w):
            is_minima = True
            current_pixel = img[row, col]

            # Check 4-connected neighbors
            for row_id in range(row - 1, row + 2):
                if row_id < 0 or row_id >= h:
                    continue
                for col_id in range(col - 1, col + 2):
                    if col_id < 0 or col_id >= w:
                        continue
                    if row_id == row and col_id == col:
                        continue
                    if is_4connected(row, col, row_id, col_id):
                        if img[row_id, col_id] < current_pixel:
                            is_minima = False
                            break
                if not is_minima:
                    break

            # If the pixel is a local minimum, assign it a label
            if is_minima:
                markers[row, col] = current_label
                current_label += 1

    return markers

def iterativeMinFollowing(img, markers):
    """
    Parameters:
    img - input image
    markers - initial markers from getRegionalMinima

    Returns:
    markers_copy - final labeled image
    """
    markers_copy = np.copy(markers)
    h, w = img.shape

    while True:
        n_unmarked_pix = 0  # Counter for unmarked pixels

        for row in range(h):
            for col in range(w):
                if markers_copy[row, col] != 0:
                    continue  # Skip already labeled pixels

                # Find the smallest 4-connected neighbor
                min_row, min_col = getSmallestNeighborIndex(img, row, col)

                # If the smallest neighbor has a label, copy it
                if markers_copy[min_row, min_col] != 0:
                    markers_copy[row, col] = markers_copy[min_row, min_col]
                else:
                    n_unmarked_pix += 1  # Increment counter if still unlabeled

        print('n_unmarked_pix:', n_unmarked_pix)

        # Stop if all pixels are labeled
        if n_unmarked_pix == 0:
            break

    return markers_copy
