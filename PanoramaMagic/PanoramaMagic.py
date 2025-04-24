# PanoramaMagic - Personal Rebranding of Image Processing Project
# This script includes four parts:
# Part 1: Bayer pattern interpolation to reconstruct RGB image
# Part 2: Dithering using Floyd-Steinberg algorithm with KMeans-generated palette
# Part 3: Geometric transformations (rotate, scale, skew) and warping
# Part 4: Panorama stitching using ORB, RANSAC, and homography estimation

import os
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from skimage import io, color, img_as_float, transform
from sklearn.cluster import KMeans
from scipy import spatial
from skimage.color import gray2rgb, rgb2gray
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage.color import rgb2gray
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage import feature

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성


# --- PART 1: Bayer Interpolation ---
def part1():
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'
    filename_gridR = 'gridR.bmp'
    filename_gridG = 'gridG.bmp'
    img = io.imread(filename_Grayimage, as_gray =True)

    h,w = img.shape

    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h,w,3),np.uint8)

    # reconstruction of the IG channel IG
    IG = np.copy(img) # copy the image into each channel
    # For red and blue channels, initialize as zeros.
    IR = np.copy(img)
    IB = np.zeros((h, w), np.float32)

    # Process the image in 4x4 blocks.
    # We assume that the Bayer pattern follows a 2x2 repetition:
    # Row0: G, R, G, R
    # Row1: B, G, B, G
    # Row2: G, R, G, R
    # Row3: B, G, B, G
    #
    # For clarity, label each 4x4 block as:
    #   A   B   C   D
    #   E   F   G   H
    #   I   J   K   L
    #   M   N   O   P
    #
    # Valid sensor readings (from the Bayer pattern) are:
    #   Green at: A, C, F, H, I, K, N, P
    #   Red   at: B, D, J, L
    #   Blue  at: E, G, M, O
    #
    # 나머지 위치는 주변 유효 픽셀의 평균으로 보간합니다.
    for row in range(0, h, 4):
        for col in range(0, w, 4):
            # --- GREEN CHANNEL (IG) ---
            # (0,1) -> B: interpolate from A, C, F
            IG[row, col+1] = (int(img[row, col]) + int(img[row, col+2]) + int(img[row+1, col+1]))/3
            # (0,3) -> D: interpolate from C, H
            IG[row, col+3] = (int(img[row, col+2]) + int(img[row+1, col+3]))/2
            # (1,0) -> E: interpolate from A, F, I
            IG[row+1, col] = (int(img[row, col]) + int(img[row+1, col+1]) + int(img[row+2, col]))/3
            # (1,2) -> G: interpolate from C, F, H, K
            IG[row+1, col+2] = (int(img[row, col+2]) + int(img[row+1, col+1]) + int(img[row+1, col+3]) + int(img[row+2, col+2]))/4
            # (2,1) -> J: interpolate from I, K, F, N
            IG[row+2, col+1] = (int(img[row+2, col]) + int(img[row+2, col+2]) + int(img[row+1, col+1]) + int(img[row+3, col+1]))/4
            # (2,3) -> L: interpolate from K, H, P
            IG[row+2, col+3] = (int(img[row+2, col+2]) + int(img[row+1, col+3]) + int(img[row+3, col+3]))/3
            # (3,0) -> M: interpolate from I, N  (as given in template)
            IG[row+3, col] = (int(img[row+2, col]) + int(img[row+3, col+1]))/2
            # (3,2) -> O: interpolate from N, P
            IG[row+3, col+2] = (int(img[row+3, col+1]) + int(img[row+3, col+3]))/2

            # --- RED CHANNEL (IR) ---
            # Valid red pixels:
            IR[row, col+1] = img[row, col+1]      # B
            IR[row, col+3] = img[row, col+3]        # D
            IR[row+2, col+1] = img[row+2, col+1]    # J
            IR[row+2, col+3] = img[row+2, col+3]    # L

            # Interpolate missing red pixels:
            # First row:
            IR[row, col] = IR[row, col+1]                      # A: copy from B (first column rule)
            IR[row, col+2] = (int(IR[row, col+1]) + int(IR[row, col+3]))/2   # C: average of B and D

            # Second row:
            IR[row+1, col+1] = (int(IR[row, col+1]) + int(IR[row+2, col+1]))/2  # between B and J
            IR[row+1, col+3] = (int(IR[row, col+3]) + int(IR[row+2, col+3]))/2  # between D and L
            IR[row+1, col] = IR[row+1, col+1]                 # first column: copy from col+1
            IR[row+1, col+2] = (int(IR[row+1, col+1]) + int(IR[row+1, col+3]))/2  # average horizontally

            # Third row:
            IR[row+2, col] = IR[row+2, col+1]                 # I: copy from J
            IR[row+2, col+2] = (int(IR[row+2, col+1]) + int(IR[row+2, col+3]))/2  # K: average of J and L

            # Fourth row (last row): copy entire row from the third row (second last row)
            for k in range(4):
                IR[row+3, col+k] = IR[row+2, col+k]

            # ------------------------------
            # 3) reconstruction of the IB channel (Blue)
            #    - last column & first row empty => copy
            # ------------------------------

            # valid blue pixels: E(row+1, col), G(row+1, col+2), M(row+3, col), O(row+3, col+2)
            IB[row+1, col]   = img[row+1, col]     # E
            IB[row+1, col+2] = img[row+1, col+2]   # G
            IB[row+3, col]   = img[row+3, col]     # M
            IB[row+3, col+2] = img[row+3, col+2]   # O

            # 내부 보간:
            # F(row+1, col+1) = (E+G)/2
            IB[row+1, col+1] = (IB[row+1, col] + IB[row+1, col+2]) / 2.0
            # H(row+1, col+3) => 나중에 마지막 열 복사하므로 여기선 무시 or 임시
            # I(row+2, col)   = (E+M)/2
            IB[row+2, col] = (IB[row+1, col] + IB[row+3, col]) / 2.0
            # J(row+2, col+1) = (E+G+M+O)/4
            IB[row+2, col+1] = (
                IB[row+1, col]   +  # E
                IB[row+1, col+2] +  # G
                IB[row+3, col]   +  # M
                IB[row+3, col+2]    # O
            ) / 4.0
            # K(row+2, col+2) = (G+O)/2
            IB[row+2, col+2] = (IB[row+1, col+2] + IB[row+3, col+2]) / 2.0
            # N(row+3, col+1) = (M+O)/2
            IB[row+3, col+1] = (IB[row+3, col] + IB[row+3, col+2]) / 2.0
            # P(row+3, col+3) => 마지막 열에서 복사

            # (1) 마지막 열(col+3)은 전부 (col+2)에서 복사
            for r2 in range(4):
                IB[row+r2, col+3] = IB[row+r2, col+2]

            # (2) 첫 번째 행(row+0)은 전부 (row+1)에서 복사
            for c2 in range(4):
                IB[row, col+c2] = IB[row+1, col+c2]

    # Merge channels into final RGB image in the order: Red, Green, Blue
    rgb[:, :, 0] = IR
    rgb[:, :, 1] = IG
    rgb[:, :, 2] = IB

    # plotting code
    plt.imshow(IG, cmap='gray'),plt.title('IG')
    plt.savefig(os.path.join(output_dir, "Bayer_Green.png"))
    plt.show()
    
    plt.imshow(IR, cmap='gray'),plt.title('IR')
    plt.savefig(os.path.join(output_dir, "Bayer_Red.png"))
    plt.show()

    plt.imshow(IB, cmap='gray'),plt.title('IB')
    plt.savefig(os.path.join(output_dir, "Bayer_Blue.png"))
    plt.show()

    plt.imshow(rgb),plt.title('rgb')
    plt.savefig(os.path.join(output_dir, "Bayer_reconstruction.png"))
    plt.show()


# --- PART 2: Dithering with Dynamic Palette ---
def part2():
    # Finds the closest colour in the palette using kd-tree.
    def nearest(palette, colour):
        dist, i = palette.query(colour)
        return palette.data[i]


    # Make a kd-tree palette from the provided list of colours
    def makePalette(colours):
        #print(colours)
        return spatial.KDTree(colours)


    # Dynamically calculates and N-colour palette for the given image
    # Uses the KMeans clustering algorithm to determine the best colours
    # Returns a kd-tree palette with those colours
    def findPalette(image, nColours):
        pixels = image.reshape(-1, 3)  # Reshape image into a 2D array of pixels
        kmeans = KMeans(
            n_clusters=nColours, 
            n_init="auto", 
            random_state=0, 
            algorithm='lloyd'
        )
        kmeans.fit(pixels)
        colours = kmeans.cluster_centers_

        print("colours:")
        print(colours)

        return makePalette(colours)

    def ModifiedFloydSteinbergDitherColor(image, palette):
        h, w, _ = image.shape

        for y in range(h):
            for x in range(w):
                oldpixel = image[y, x].copy()
                newpixel = nearest(palette, oldpixel)
                image[y, x] = newpixel
                quant_error = oldpixel - newpixel

                # Distribute the quantization error to neighboring pixels
                if x + 1 < w:
                    image[y, x+1] = np.clip(image[y, x+1] + quant_error * (11/26), 0, 1)
                if y + 1 < h:
                    if x - 1 >= 0:
                        image[y+1, x-1] = np.clip(image[y+1, x-1] + quant_error * (5/26), 0, 1)
                    image[y+1, x] = np.clip(image[y+1, x] + quant_error * (7/26), 0, 1)
                    if x + 1 < w:
                        image[y+1, x+1] = np.clip(image[y+1, x+1] + quant_error * (3/26), 0, 1)
        return image

    nColours = 7 # The number colours: change to generate a dynamic palette
    imfile = 'mandrill.png'
    image = io.imread(imfile)
    orig = image.copy()

    # Strip the alpha channel if it exists
    image = image[:,:,:3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    colours = img_as_float([colours.astype(np.ubyte)])[0]

    img = ModifiedFloydSteinbergDitherColor(image, palette)


    plt.imshow(img),plt.title(f'Dithered Image (nColours = {nColours})')
    plt.savefig(os.path.join(output_dir, "Dithered_mandrill.png"))
    plt.show()



# --- PART 3: Geometric Transformations ---
def part3():
    def read_image():
        original_img = io.imread('bird.jpeg')
        return img_as_float(original_img)  # Float 변환

    def calculate_trans_mat(image):
        """ Return translation matrix that shifts center of image to the origin and its inverse """
        h, w = image.shape[:2]
        tx, ty = w / 2, h / 2  # 이미지 중심

        trans_mat = np.array([
            [1, 0, -tx],
            [0, 1, -ty],
            [0, 0, 1]
        ])

        trans_mat_inv = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        return trans_mat, trans_mat_inv

    def apply_transform(image, transform_matrix):
        """ Apply affine transformation manually (No built-in functions) """
        h, w = image.shape[:2]
        out_img = np.zeros_like(image)

        inv_matrix = np.linalg.inv(transform_matrix)  # 역행렬 계산

        for out_y in range(h):
            for out_x in range(w):
                coord = np.array([out_x, out_y, 1])
                x_in, y_in, _ = inv_matrix @ coord
                x_in, y_in = int(x_in), int(y_in)

                if 0 <= x_in < w and 0 <= y_in < h:
                    out_img[out_y, out_x] = image[y_in, x_in]

        return out_img

    def rotate_image(image):
        ''' Rotate and return image (75 degrees counter-clockwise) '''
        trans_mat, trans_mat_inv = calculate_trans_mat(image)

        angle = 75  # 75도 반시계 방향 (Counter-clockwise)
        angle_rad = np.radians(angle)

        Tr = np.array([
            [ np.cos(angle_rad), np.sin(angle_rad), 0],
            [-np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

        Tr = trans_mat_inv @ Tr @ trans_mat  # 중심 기준 회전 적용
        return apply_transform(image, Tr), Tr

    def scale_image(image):
        ''' Scale image and return '''
        trans_mat, trans_mat_inv = calculate_trans_mat(image)

        Ts = np.array([
            [1.5, 0, 0],
            [0, 2.5, 0],
            [0, 0, 1]
        ])

        Ts = trans_mat_inv @ Ts @ trans_mat
        return apply_transform(image, Ts), Ts

    def skew_image(image):
        ''' Skew image and return '''
        trans_mat, trans_mat_inv = calculate_trans_mat(image)

        Tskew = np.array([
            [1, 0.2, 0],
            [0.2, 1, 0],
            [0, 0, 1]
        ])

        Tskew = trans_mat_inv @ Tskew @ trans_mat
        return apply_transform(image, Tskew), Tskew

    def combined_warp(image):
        ''' Combine rotation, scaling, and skewing transformations and return image + matrix '''
        trans_mat, trans_mat_inv = calculate_trans_mat(image)

        Tr = np.array([
            [np.cos(np.radians(75)), np.sin(np.radians(75)), 0],
            [-np.sin(np.radians(75)), np.cos(np.radians(75)), 0],
            [0, 0, 1]
        ])

        Ts = np.array([
            [1.5, 0, 0],
            [0, 2.5, 0],
            [0, 0, 1]
        ])

        Tskew = np.array([
            [1, 0.2, 0],
            [0.2, 1, 0],
            [0, 0, 1]
        ])

        T_final = trans_mat_inv @ (Ts @ Tr @ Tskew) @ trans_mat  

        return apply_transform(image, T_final), T_final

    def combined_warp_bilinear(image):
        ''' Perform the combined warp with bilinear interpolation (Part 5, using skimage) '''
        _, T_final = combined_warp(image)  # 변환 행렬 가져오기
        T_inv = np.linalg.inv(T_final)  # 역변환 행렬

        def transform_coords(coords):
            x, y = coords.T
            ones = np.ones_like(x)
            coords_homogeneous = np.vstack((x, y, ones))  # Homogeneous coordinates
            transformed_coords = T_inv @ coords_homogeneous  # 변환 적용
            return transformed_coords[:2].T  # (x, y)만 반환

        out_img = warp(image, transform_coords, order=1, mode='constant', cval=0, clip=True)

        return out_img

    # Plotting code here
    image = read_image()
    plt.imshow(image), plt.title("Original Image"), 

    rotated_img, _ = rotate_image(image)
    plt.imshow(rotated_img), plt.title("Rotated Image")
    plt.savefig(os.path.join(output_dir, "Rotated_Image.png"))
    plt.show()

    scaled_img, _ = scale_image(image)
    plt.imshow(scaled_img), plt.title("Scaled Image")
    plt.savefig(os.path.join(output_dir, "Scaled_Image.png"))
    plt.show()
    
    skewed_img, _ = skew_image(image)
    plt.imshow(skewed_img), plt.title("Skewed Image"), 
    plt.savefig(os.path.join(output_dir, "Skewed_Image.png")), 
    plt.show()
    

    combined_warp_img, _ = combined_warp(image)
    plt.imshow(combined_warp_img), plt.title("Combined Warp Image")
    plt.savefig(os.path.join(output_dir, "Combined_Warp_Image.png")), 
    plt.show()

    combined_warp_bilinear_img = combined_warp_bilinear(image)
    plt.imshow(combined_warp_bilinear_img), plt.title("Combined Warp Image with Bilinear Interpolation"), 
    plt.savefig(os.path.join(output_dir, "Combined_Warp_Image_with_Bilinear_Interpolation.png")),
    plt.show() 

# --- PART 4: Panorama Stitching ---
def part4():
    # Seed for reproducibility
    np.random.seed(100)

    # Load images
    filename1 = 'im1.jpg'
    filename2 = 'im2.jpg'

    image0 = io.imread(filename1, as_gray=True)
    image1 = io.imread(filename2, as_gray=True)

    # Display original images
    plt.figure(figsize=(8,10))    
    plt.imshow(image0, cmap='gray'), plt.title("First Image")
    plt.imshow(image1, cmap='gray'), plt.title("Second Image")

    # -------- Feature detection and matching -----
    # Initialize ORB detector
    orb = ORB(n_keypoints=1000, fast_threshold=0.05)

    # Detect keypoints and descriptors in the first image
    orb.detect_and_extract(image0)
    keypoints1, descriptors1 = orb.keypoints, orb.descriptors

    # Detect keypoints and descriptors in the second image
    orb.detect_and_extract(image1)
    keypoints2, descriptors2 = orb.keypoints, orb.descriptors

    # Match descriptors using Brute-Force matcher with ratio test
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True, max_ratio=0.8, max_distance=0.8)

    # -------- Transform estimation -------
    # Prepare source and destination points for RANSAC
    # Swap row and column indices (y, x) -> (x, y)
    src = keypoints1[matches12[:, 0]][:, ::-1]  # Swap to (x, y)
    dst = keypoints2[matches12[:, 1]][:, ::-1]  # Swap to (x, y)

    # Compute homography using RANSAC
    model_robust, inliers = ransac(
        (dst, src),
        ProjectiveTransform,
        min_samples=4,
        residual_threshold=2,
        max_trials=5000
    )

    # ------------- Warping ----------------
    # Get the shape of the second image
    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])

    # Apply transformation with corrected offset
    offset = SimilarityTransform(translation=-corner_min)
    image0_ = warp(image0, offset.inverse,
                  output_shape=output_shape)

    image1_ = warp(image1, (model_robust + offset).inverse,
                  output_shape=output_shape)

    # Display warped images
    plt.imshow(image0_, cmap="gray"), plt.title("Warped first image")
    plt.savefig(os.path.join(output_dir, "Warped_first_Image.png"))
    plt.imshow(image1_, cmap="gray"), plt.title("Warped second image")
    plt.savefig(os.path.join(output_dir, "Warped_second_Image.png"))
    plt.show()

    # Add alpha channel to the warped images
    def add_alpha(image, background=-1):
        """Add an alpha layer to the image."""
        rgb = color.gray2rgb(image)
        alpha = (image != background)
        return np.dstack((rgb, alpha))

    image0_alpha = add_alpha(image0_)
    image1_alpha = add_alpha(image1_)

    # Merge images
    merged = image0_alpha + image1_alpha
    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]

    # Normalize merged image to [0, 1] range
    merged = np.clip(merged, 0, 1)

    # Display the final stitched image
    plt.figure(figsize=(8,10))
    plt.imshow(merged, cmap="gray")
    plt.imsave('imgOut.png', merged)
    plt.savefig(os.path.join(output_dir, "Merged_Panorama.png"))
    plt.show()

    # Plot 10 random inlier matches
    inlier_matches = matches12[inliers]
    fig = plt.figure(figsize=[10, 10])
    plot_matches(
        plt.gca(),
        io.imread(filename1, as_gray=False),
        io.imread(filename2, as_gray=False),
        keypoints1,
        keypoints2,
        inlier_matches[np.random.choice(inlier_matches.shape[0], 10, replace=False)],
        only_matches=True
    )
    plt.savefig(os.path.join(output_dir, "Inlier.png"))

    plt.show()

if __name__ == "__main__":
    # You can comment those lines to run only part of the code you are working on but remember to uncomment before submission
    print("***************Output for part 1:")
    part1()
    print("***************Output for part 2:")
    part2()
    print("***************Output for part 3:")
    part3()
    print("***************Output for part 4:")
    part4()