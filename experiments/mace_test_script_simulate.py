import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import time
from skimage.restoration import denoise_bilateral
import cv2 as cv


# add parent path to import functions in the comiser folder 
import sys
sys.path.append('../comiser')  
import pnp_utils as pnp
import utils as cu
import img_utils as cimgu

# # Example of a proximal operator function G_mu, implementing soft thresholding
# def G_mu(x, mu):
#     # Soft thresholding as an example of a proximal map
#     return np.sign(x) * np.maximum(np.abs(x) - mu, 0)

# Define the dimensions of the problem
N = 3  # Number of dimensions, or the number of frames
mu = 0.1
rho = 0.7  # Step size or regularization parameter try 0.7 or 0.8 


# Main iterative process
max_iterations = 25
tolerance = 1e-3

image_size = 256                # Image size
P = 10                           # Blur kernel with size (2P+1)x(2P+1)
filter_std = 2.0                # spatial standard deviation of blur kernel
decimation_rate = 2             # Integer decimation rate
lambda_param = 0.8              # Seems to become numerically unstable for lambda_param < 0.5

# Load in the 1951 AF target
file_path = 'data/USAF-1951.svg.png'
image = cu.read_png(file_path)[:, :, 1]
print(f'original image.shape: {image.shape}')

# Resize image so it is reasonable to work with
image = cu.resize_image(image, new_shape=(image_size, image_size))
gt_image = jnp.array(image)
print(f'ground truth image shape: {gt_image.shape}')
cv.imwrite('./data/gt_image.png', image*255)


# Generate a gaussian kernel
kernel = pnp.gen_gaussian_filter(P, 2.0)
measured_image = pnp.apply_G(gt_image, kernel, decimation_rate)

# Check data range
print(f"Data range: {measured_image.min()} to {measured_image.max()}")

kernels = np.expand_dims(kernel, axis=0)
measured_images = np.expand_dims(measured_image, axis=0)
gt_image_3dim = np.expand_dims(gt_image, axis=0)
gt_images = np.expand_dims(gt_image, axis=0)


rad = 2
frameNumber = (rad+1)**2   # may increase the numebr of frames
#for fn in range(frameNumber):
    #shiftx = (np.random.rand() - 0.5) * 0.5
    #shifty = (np.random.rand() -0.5) * 0.5

fn = 0
pre_kernel = np.expand_dims(kernel, axis=0)

for i in range (-rad, rad+1):
    for j in range(-rad,rad+1):
        shiftx = i / rad / 2 * decimation_rate
        shifty = j / rad / 2 * decimation_rate

        #shiftx = (np.random.rand() - 0.5) * 0.5
        #shifty = (np.random.rand() -0.5) * 0.5

        print(f"shift {fn, shiftx, shifty} ")

        kernel_shift = cimgu.fft_subpixel_shift(kernel, shiftx, shifty)
        measured_image_shift = pnp.apply_G(gt_image, kernel_shift, decimation_rate)

        # # add noise
        measured_image_shift = cu.add_noise_to_image(measured_image_shift, 0, 0.1)
        measured_image_shift = np.clip(measured_image_shift, 0, 1)
        
        # save the images
        variable_part = f"image_{fn}"
        file_extension = ".png"
        output_path = f"./data/{variable_part}{file_extension}"
        cv.imwrite(output_path, measured_image_shift*255)

        # Stack Gaussian kernels into a 3D array
        this_kernel = np.expand_dims(kernel_shift, axis=0)
        stacked_kernels = np.concatenate((pre_kernel, this_kernel), axis=0) # Stack along the third dimension

        print(stacked_kernels.shape)
        fn = fn + 1
        pre_kernel = stacked_kernels


        # if (j==0):
        #     cu.display_3images(gt_image, measured_image, measured_image_shift,  title1='GT', title2='measured_0', title3='measured shifted')

        # kernel_shift = np.expand_dims(kernel_shift, axis=0)
        # measured_image_shift = np.expand_dims(measured_image_shift, axis=0)

        # # stack 
        # kernels = np.concatenate((kernels, kernel_shift), axis=0)
        # measured_images = np.concatenate((measured_images, measured_image_shift), axis=0)
        # gt_images = np.concatenate((gt_images, gt_image_3dim), axis=0)


stacked_kernels = stacked_kernels[1:]
np.save('data/kernels', stacked_kernels)




print("Shape of combined image array:", measured_images.shape)
print("Shape of combined kernel array:", kernels.shape)


