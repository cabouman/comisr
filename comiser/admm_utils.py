
import numpy as np
import matplotlib.pyplot as plt
#import comiser.pnp_utils as pnp 
#import comiser.utils as cu


import numpy as np
from skimage import restoration, util, img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from bm3d import bm3d



# Denoise the image using BM4D
def my_BM3D(image_noisy):
    image_noisy = img_as_float(image_noisy)
    image_denoised = bm3d.bm3d(image_noisy, sigma=0.1)
    return image_denoised


def simple_bm3d(image, sigma):
    # Use Non-Local Means denoising without the 'multichannel' parameter
    denoised = restoration.denoise_nl_means(image,
                                            h=1.15 * sigma,
                                            fast_mode=True,
                                            patch_size=7,
                                            patch_distance=11)
    return denoised


import imageio
from scipy.fftpack import dct, idct
from skimage.util import view_as_windows, view_as_blocks

import cvxpy as cp
def total_variation_denoise(noisy_image, lambda_value, num_iterations):
    # Initialize denoised image with noisy image
    denoised_image = np.copy(noisy_image)
    
    # Define the TV denoising problem
    image_shape = noisy_image.shape
    denoised_image_var = cp.Variable(image_shape)
    objective = cp.Minimize(cp.norm(denoised_image_var - noisy_image, 'fro') ** 2 / 2 + lambda_value * cp.tv(denoised_image_var))
    problem = cp.Problem(objective)
    
    # Solve the problem iteratively
    for i in range(num_iterations):
        problem.solve()
        denoised_image = denoised_image_var.value
    
    return denoised_image

# # Example usage
# # Assuming 'noisy_image' is your input noisy image
# # 'lambda_value' is the trade-off parameter controlling smoothness vs fidelity
# # 'num_iterations' is the number of iterations for gradient descent
# denoised_image = total_variation_denoise(noisy_image, lambda_value=0.1, num_iterations=100)


import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter

def compute_similarity(patch1, patch2, h):
    # Compute squared differences between corresponding pixels
    squared_diff = (patch1 - patch2) ** 2
    # Compute the Gaussian weighted sum of squared differences
    weighted_squared_diff = gaussian_filter(squared_diff, sigma=h)
    # Compute the similarity between the patches
    similarity = np.exp(-weighted_squared_diff)
    return similarity

def nlm_denoise(image, h, sigma, patch_size, search_window_size):
    # Pad the image to handle edge cases
    padded_image = np.pad(image, patch_size, mode='symmetric')
    
    # Initialize denoised image
    denoised_image = np.zeros_like(image)
    
    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the patch centered around the current pixel
            patch_center = (i + patch_size, j + patch_size)
            patch = padded_image[patch_center[0]-patch_size:patch_center[0]+patch_size+1,
                                 patch_center[1]-patch_size:patch_center[1]+patch_size+1]
            
            # Define search window boundaries
            row_start = max(0, patch_center[0] - search_window_size)
            row_end = min(padded_image.shape[0], patch_center[0] + search_window_size)
            col_start = max(0, patch_center[1] - search_window_size)
            col_end = min(padded_image.shape[1], patch_center[1] + search_window_size)
            
            # Initialize weighted average and normalization factor
            weighted_sum = 0
            normalization_factor = 0
            
            # Iterate over each pixel in the search window
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    # Extract the patch centered around the current pixel in the search window
                    search_patch_center = (row, col)
                    search_patch = padded_image[search_patch_center[0]-patch_size:search_patch_center[0]+patch_size+1,
                                                 search_patch_center[1]-patch_size:search_patch_center[1]+patch_size+1]
                    
                    # Compute the similarity between the patches
                    similarity = compute_similarity(patch, search_patch, h)
                    
                    # Update the weighted sum and normalization factor
                    weighted_sum += similarity * padded_image[row, col]
                    normalization_factor += similarity
            
            # Compute the denoised pixel value
            denoised_image[i, j] = weighted_sum / normalization_factor
    
    return denoised_image

# # Example usage
# # Assuming 'noisy_image' is your input noisy image
# # 'h' is the smoothing parameter controlling the level of filtering
# # 'sigma' is the standard deviation of the Gaussian filter applied to the squared differences
# # 'patch_size' is the size of the patches used for computing similarities
# # 'search_window_size' is the size of the search window for finding similar patches
# denoised_image = nlm_denoise(noisy_image, h=0.1, sigma=1.0, patch_size=3, search_window_size=10)



def bm3d_1st_step(image, block_size=8):
    # Pad image to handle edge blocks
    pad_width = block_size // 2
    padded_image = np.pad(image, pad_width, mode='reflect')
    
    # Number of blocks calculation needs to consider the image size
    num_blocks_h = (image.shape[0] - block_size) // block_size + 1
    num_blocks_w = (image.shape[1] - block_size) // block_size + 1

    # Initialize arrays to store filtered blocks
    filtered_blocks = np.zeros((num_blocks_h, num_blocks_w, block_size, block_size))
    weight_sum = np.zeros_like(image)
    denoised_image = np.zeros_like(image)

    # Iterate over all possible positions to place a block
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            row_start = i * block_size
            col_start = j * block_size
            row_end = row_start + block_size
            col_end = col_start + block_size

            # Check if the block is fully within the image boundaries
            if row_end <= image.shape[0] and col_end <= image.shape[1]:
                block = padded_image[row_start:row_end, col_start:col_end]
                filtered_blocks[i, j] = block  # Assume some filtering happens

                # Aggregation
                denoised_image[row_start:row_end, col_start:col_end] += filtered_blocks[i, j]
                weight_sum[row_start:row_end, col_start:col_end] += 1

    # Normalize the image by the accumulated weights to average overlapping areas
    weight_sum[weight_sum == 0] = 1  # Avoid division by zero
    denoised_image /= weight_sum

    return denoised_image

def bm3d_test(measured_image):

    # Normalize the image to [0, 1] (BM3D expects float32 images in this range)
    noisy_image = measured_image.astype(np.float32) / 255.0

    # Add synthetic Gaussian noise
    # noisy_image = image + 0.1 * np.random.normal(loc=0, scale=1, size=image.shape)

    # Denoise the image using BM3D
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.1, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    return denoised_image




def admm_with_proximal_backup(x, y, kernel, decimation_rate, lambda_param, rho, mu, max_iter=1000, tol=1e-4):
    """
    ADMM for solving:
    minimize_x f(x)+ rho |x-(v-u)|^2 using the proximal map.
    f(x) = |y-Gx|^2
    
    Parameters:
    x: prox_input_image
    y: measured_image
    rho : float
        Penalty parameter for the ADMM.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for the stopping criterion.
    """
    M, N = y.shape

    m = M * decimation_rate
    n = N * decimation_rate

    u = np.zeros((m,n))
    v = np.zeros((m,n))
    

    for iteration in range(max_iter):

        # x-update (solving the quadratic subproblem)
        x_tilde = v - u 
        x_old = x_tilde.copy()
        x = pnp.proximal_map_numerically_stable(x_tilde, y, kernel, decimation_rate, lambda_param)

        # Test that iterated prox is converging to the correct solution
        nrmse1 = pnp.get_nrmse_convergence_error(y, x, kernel, decimation_rate)
        print(f'RMSE 1 = {nrmse1}')

        cu.display_images(y, x, title1='Measured Image', title2=f'{iteration} Iterations of Proximal Map')


        # z-update (applying the proximal map for the L1 norm)
        v_tilde = x + u
        #v = soft_thresholding(v_tilde, mu / rho)

        # apply linear filter
        v = pnp.filter_2D_jax(v_tilde, kernel)
        

            
        # u-update (dual variable update)
        u += (x-v)
        
        # Convergence check
        if np.linalg.norm(x - x_old) < tol:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break

    return x


def soft_thresholding(v, lambda_):
    """ Soft thresholding operator for L1 regularization. """
    return np.sign(v) * np.maximum(np.abs(v) - lambda_, 0)