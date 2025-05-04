import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import io
from scipy.fftpack import dct
from scipy.fftpack import idct

from jpeg_utils import zigzag
from jpeg_utils import inverse_zigzag

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성


# TODO: Implement the function below
def convert_rgb_to_ycbcr(rgb_img):
    """
    Transform the RGB image to a YCbCr image
    """
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.168736, -0.331264, 0.5],
                                 [0.5, -0.418688, -0.081312]])
    
    shift = np.array([0, 128, 128])
    
    # (H, W, 3) → (H*W, 3)
    flat_img = rgb_img.reshape(-1, 3)
    ycbcr = np.dot(flat_img, transform_matrix.T) + shift
    ycbcr = ycbcr.reshape(rgb_img.shape)
    
    return ycbcr

# TODO: Implement the function below
def convert_ycbcr_to_rgb(ycbcr_img):
    """
    Transform the YCbCr image to a RGB image
    """
    inv_matrix = np.array([[1.0, 0.0, 1.402],
                           [1.0, -0.344136, -0.714136],
                           [1.0, 1.772, 0.0]])
    
    shift = np.array([0, 128, 128])
    flat_img = ycbcr_img.reshape(-1, 3)
    rgb = np.dot(flat_img - shift, inv_matrix.T)
    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.reshape(ycbcr_img.shape)
    return rgb

# TODO: Implement the function below
def dct2D(input_img):
    """
    Function to compute 2D Discrete Cosine Transform (DCT)
    """
    # Apply DCT with type 2 and 'ortho' norm parameters
    return dct(dct(input_img.T, type=2, norm='ortho').T, type=2, norm='ortho')

# TODO: Implement the function below
def idct2D(input_dct):
    """
    Function to compute 2D Inverse Discrete Cosine Transform (IDCT)
    """
    # Apply IDCT with type 2 and 'ortho' norm parameters
    return idct(idct(input_dct.T, type=2, norm='ortho').T, type=2, norm='ortho')



def jpeg_encode():
    """
    JPEG Encoding
    """

    # NOTE: Defining block size
    block_size = 8

    # TODO: Read image using skimage.io
    ###### Your code here ######
    img = io.imread('bird.jpg')

    plt.imshow(img)
    plt.title('input image (RGB)')
    plt.savefig(os.path.join(output_dir, "input_image_(RGB).png"))
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from RGB space to YCbCr space
    img = convert_rgb_to_ycbcr(img)
    
    plt.imshow(np.uint8(img))
    plt.title('input image (YCbCr)')
    plt.savefig(os.path.join(output_dir, "input_image_(YCbCr).png"))
    plt.axis('off')
    plt.show()

    # TODO: Get size of input image (h, w, c)
    ###### Your code here ######
    h, w, c = img.shape

    # TODO: Compute number of blocks (of size 8-by-8), cast the numbers to int
    nbh = int(np.ceil(h / block_size))
    nbw = int(np.ceil(w / block_size))

    # TODO: (If necessary) Pad the image, get size of padded image
    nbh = int(np.ceil(h / block_size))
    nbw = int(np.ceil(w / block_size))

    # TODO: Create a numpy zero matrix with size of H,W,3 called padded img
    H = nbh * block_size
    W = nbw * block_size
    padded_img = np.zeros((H, W, 3))

    # TODO: Copy the values of img into padded_img[0:h,0:w,:]
    padded_img[0:h, 0:w, :] = img

    # TODO: Display padded image
    plt.imshow(np.uint8(padded_img))
    plt.title('input padded image')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "input_padded_image.png"))
    plt.show()


    # Quantization matrices
    quantization_matrix_Y = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    quantization_matrix_CbCr = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]
    ])


    # TODO: Initialize variables for compression calculations (only for the Y channel)
    before_nonzeros = 0
    after_nonzeros = 0

    # NOTE: Iterate over blocks
    for i in range(nbh):
        
        # Compute start and end row indices of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size
        
        for j in range(nbw):
            
            # Compute start and end column indices of the block
            col_ind_1 = j * block_size 
            col_ind_2 = col_ind_1 + block_size
            
            # TODO: Select current block to process using calculated indices (through slicing)
            Yblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0]
            Cbblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1]
            Crblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2]
            
            # TODO: Apply dct2d() to selected block             
            YDCT = dct2D(Yblock)
            CbDCT = dct2D(Cbblock)
            CrDCT = dct2D(Crblock)

            # TODO: Quantization
            # Divide each element of DCT block by corresponding element in quantization matrix
            quantized_YDCT = np.round(YDCT / quantization_matrix_Y)
            quantized_CbDCT = np.round(CbDCT / quantization_matrix_CbCr)
            quantized_CrDCT = np.round(CrDCT / quantization_matrix_CbCr)

            # TODO: Reorder DCT coefficients into block (use zigzag function)
            reordered_Y = zigzag(quantized_YDCT)
            reordered_Cb = zigzag(quantized_CbDCT)
            reordered_Cr = zigzag(quantized_CrDCT)

            # TODO: Reshape reordered array to 8-by-8 2D block
            reshaped_Y = inverse_zigzag(reordered_Y, 8, 8)
            reshaped_Cb = inverse_zigzag(reordered_Cb, 8, 8)
            reshaped_Cr = inverse_zigzag(reordered_Cr, 8, 8)

            # TODO: Copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0] = reshaped_Y
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1] = reshaped_Cb
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2] = reshaped_Cr

            # TODO: Compute pixel locations with non-zero values before and after quantization (only in Y channel)
            # TODO: Compute total number of pixels
            before_nonzeros += np.count_nonzero(YDCT)
            after_nonzeros += np.count_nonzero(quantized_YDCT)

    plt.imshow(np.uint8(padded_img))
    plt.title('encoded image')
    plt.savefig(os.path.join(output_dir, "encoded_image.png"))
    plt.axis('off')
    plt.show()

    # TODO: Calculate percentage of pixel locations with non-zero values before and after to measure degree of compression 
    before_compression = (before_nonzeros / (nbh * nbw * block_size * block_size)) * 100
    after_compression = (after_nonzeros / (nbh * nbw * block_size * block_size)) * 100

    # Print statements as shown in eClass
    print('Percentage of non-zero elements in Luma channel:')
    print('Before compression: ', before_compression, '%')
    print('After compression: ', after_compression, '%')


    # Writing h, w, c, block_size into a .txt file
    np.savetxt('size.txt', [h, w, c, block_size])

    # Writing the encoded image into a file
    np.save('encoded.npy', padded_img)


def jpeg_decode():
    """
    JPEG Decoding
    """

    # TODO: Load 'encoded.npy' into padded_img (using np.load() function)
    ###### Your code here ######
    padded_img = np.load('encoded.npy')

    # TODO: Load h, w, c, block_size and padded_img from the size.txt file
    ###### Your code here ######
    h, w, c, block_size = np.loadtxt('size.txt')

    # TODO: 6. Get size of padded_img, cast to int if needed
    h, w, c, block_size = int(h), int(w), int(c), int(block_size)
    H, W, _ = padded_img.shape

    # TODO: Create the quantization matrix (Same as before)
    quantization_matrix_Y = np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ])

    quantization_matrix_CbCr = np.array([
        [17,18,24,47,99,99,99,99],
        [18,21,26,66,99,99,99,99],
        [24,26,56,99,99,99,99,99],
        [47,66,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99],
        [99,99,99,99,99,99,99,99]
    ])


    # TODO: Compute number of blocks (of size 8-by-8), cast to int
    nbh = H // block_size
    nbw = W // block_size

    # TODO: iterate over blocks
    for i in range(nbh):
        
            # Compute start and end row indices of the block
            row_ind_1 = i * block_size
            
            row_ind_2 = row_ind_1 + block_size
            
            for j in range(nbw):
                
                # Compute start and end column indices of the block
                col_ind_1 = j * block_size

                col_ind_2 = col_ind_1 + block_size
                
                Yblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0]
                Cbblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1]
                Crblock = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2]

                Yreshaped = zigzag(Yblock)
                Cbreshaped = zigzag(Cbblock)
                Crreshaped = zigzag(Crblock)

                Yreordered = inverse_zigzag(Yreshaped, 8, 8)
                Cbreordered = inverse_zigzag(Cbreshaped, 8, 8)
                Crreordered = inverse_zigzag(Crreshaped, 8, 8)

                dequantized_YDCT = Yreordered * quantization_matrix_Y
                dequantized_CbDCT = Cbreordered * quantization_matrix_CbCr
                dequantized_CrDCT = Crreordered * quantization_matrix_CbCr

                YIDCT = idct2D(dequantized_YDCT)
                CbIDCT = idct2D(dequantized_CbDCT)
                CrIDCT = idct2D(dequantized_CrDCT)

                padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 0] = YIDCT
                padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 1] = CbIDCT
                padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2, 2] = CrIDCT

    # TODO: Remove out-of-range values
    padded_img = np.clip(padded_img, 0, 255)

    plt.imshow(np.uint8(padded_img))
    plt.title('decoded padded image')
    plt.savefig(os.path.join(output_dir, "decoded_padded_image.png"))
    plt.axis('off')
    plt.show()

    # TODO: Get original sized image from padded_img
    ###### Your code here ######
    decoded_img = padded_img[0:h, 0:w, :]

    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded image')
    plt.savefig(os.path.join(output_dir, "decoded_image.png"))
    plt.axis('off')
    plt.show()
    
    # TODO: Convert the image from YCbCr to RGB
    decoded_img = convert_ycbcr_to_rgb(decoded_img)
    
    # TODO: Remove out-of-range values
    decoded_img = np.clip(decoded_img, 0, 255)
    
    plt.imshow(np.uint8(decoded_img))
    plt.title('decoded image (RGB)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "decoded_image_(RGB).png"))

    plt.show()


if __name__ == '__main__':
    jpeg_encode()
    jpeg_decode()

