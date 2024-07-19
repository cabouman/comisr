import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import time
from skimage.restoration import denoise_bilateral
import cv2 as cv
import os

# add parent path to import functions in the comiser folder 
import sys
sys.path.append('../comiser')  
import pnp_utils as pnp
import utils as cu
import img_utils as cimgu


# Load in the image
image_folder = 'data/'

def load_images_from_folder(file_path, prefix="image_", extension=".png", num_images=26):
    images = []
    for i in range(0, num_images + 1):
        filename = os.path.join(file_path, f"{prefix}{i}{extension}")
        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load image {filename}")
    return images


# Example function F, which could be a linear transformation or any application-specific function

def F(w, measured_images, kernels, decimation_rate, lambda_param):
    for j in range(w.shape[0]):

        #measured_image = measured_images[j,:]
        measured_image = measured_images[j]
        kernel = kernels[j] # use 0 to select the first kernel
        this_w = w[j]
        #cu.display_images(this_w, measured_image, title1='thisw', title2='measured')
        this_w = pnp.proximal_map_numerically_stable(this_w , measured_image, kernel, decimation_rate, lambda_param )
        #cu.display_image(this_w, title='proxi')
        w[j,:] = this_w
    return w  # Assuming A is some predefined matrix
    

# Dummy function G_mu, assuming some form of projection or transformation
def G_mu(x):
    new_array = np.ones_like(x)
    new_array = new_array * np.mean(x, axis=0)
    return new_array  # Example: Simple thresholding

# Load images
images = load_images_from_folder(image_folder, num_images=26)
kernels = np.load('data/kernels.npy')
gt_image = cv.imread('data/gt_image.png', cv.IMREAD_GRAYSCALE)

cu.display_image(images[1], title='load frame 1')


# Check the number of images loaded
print(f"Loaded {len(images)} images.")
print(f"Loaded {len(kernels)} kernels.")


# Print the shape of each loaded image
for idx, img in enumerate(images):
    print(f"Shape of image {idx + 1}: {img.shape}")


# Define the dimensions of the problem
N = 3  # Number of dimensions, or the number of frames
mu = 0.1
rho = 0.7  # Step size or regularization parameter try 0.7 or 0.8 


# Main iterative process
max_iterations = 20
tolerance = 1e-3

image_size = 256                # Image size
P = 10                          # Blur kernel with size (2P+1)x(2P+1)
filter_std = 2.0                # spatial standard deviation of blur kernel
decimation_rate = 2             # Integer decimation rate
lambda_param = 0.8              # Seems to become numerically unstable for lambda_param < 0.5

#w = np.zeros(len(images), gt_image.shape)
w = np.zeros((len(images),) + gt_image.shape)

print(w.shape)

#w = gt_images
#mu = 0.1

# MACE 

rmse_values = []
for iteration in tqdm(range(max_iterations)):
    # Step 1:
    x = F(w, images, kernels, decimation_rate, lambda_param)

    # Step 2:
    z = G_mu(2 * x - w)
    
    """ # Step 3: add denoiser
    # project data to (0,1) space
    normalized_z, min_val, max_val = cu.min_max_normalize(z)

    print('min and max:', min_val, max_val)

    # denoiser
    #denoiser_funtion = pnp.get_denoiser(method='BM3D')
    #denoised_image = denoiser_funtion(normalized_z, 0.1)

    # Apply BM3D denoising
    # Apply BM3D denoising to each slice of the 3D image
    denoised_image = np.zeros_like(z)
    denoised_image[0] = denoise_bilateral(z[0], sigma_color=0.05, sigma_spatial=0.05*(max_val - min_val))

    for i in range(1, z.shape[0]):
        denoised_image[i] = denoised_image[0] 

    #denoised_image = denoise_bilateral(z, sigma_color=0.05, sigma_spatial=15)

    
    #project data back to original space
    z = cu.min_max_denormalize(denoised_image, min_val, max_val) """

    denoiser_funtion = pnp.get_denoiser(method='GF')
    #denoiser_funtion = pnp.get_denoiser(method='BM3D')


    denoised_image = np.zeros_like(z)
    denoised_image[0] = denoiser_funtion(z[0], 0.1)
    for i in range(1, z.shape[0]):
        denoised_image[i] = denoised_image[0] 

    z = denoised_image

    # Step 4
    w_new = w + 2 * rho * (z - x)

    # Convergence check (stop if the update is small)
    if np.linalg.norm(w_new - w) < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break
    
    w = w_new
    temp = z[0,:]
    #cu.display_image(temp, title='restored')
    rmse = pnp.mse(temp, gt_image)
    rmse_values.append(rmse)
    time.sleep(0.1)


# Return the result
x_star = z[0,:]

# Save to a binary file in NumPy `.npy` format
np.save('./data/rmse_values.npy', rmse_values)

# compute the mse
rmse = pnp.mse(x_star, gt_image)
print(f"RMSE between the restored image and GT image is {rmse}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(rmse_values, marker='o', linestyle='-', color='b')
plt.title('Convergence Chart of RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.grid(True)
plt.show()


restored_image = cu.convert_jax_to_image(x_star)
restored_image.save('./data/restored_image_mace.png')
cu.display_3images(gt_image, images[0], x_star, title1='GT', title2='noised', title3='MACE restored')
