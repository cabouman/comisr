import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm
import time
from skimage.restoration import denoise_bilateral
import cv2 as cv
import os
import scipy.io
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from skimage.transform import resize
from skimage import restoration

# add parent path to import functions in the comiser folder 
import sys
sys.path.append('../comiser')  
sys.path.append('../')  

import pnp_utils as pnp
import utils as cu
import img_utils as cimgu


DenoiserEnabled = 0
# Load in the image
image_folder = 'data/eric_data/'
NUM_images = 1


def load_images_from_folder(file_path, prefix="frame_", extension=".png", num_images = 10):
    images = []
    for i in range(1, num_images + 1):
        #filename = os.path.join(file_path, f"{prefix}{i}{extension}")
        filename = os.path.join(file_path, "{}{}{}".format(prefix, i, extension))

        img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

        if img is not None:
            images.append(img)
        else:
            #print(f"Warning: Could not load image {filename}")
            print("Warning: Could not load image {}".format(filename))

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
images = load_images_from_folder(image_folder, num_images = NUM_images)
#gt_image = cv.imread(f"{image_folder}gt_image.png", cv.IMREAD_GRAYSCALE)
gt_image = cv.imread("{}gt_image.png".format(image_folder), cv.IMREAD_GRAYSCALE)


gt_image, min_val, max_val = cu.min_max_normalize(gt_image)
images, min_val, max_val = cu.min_max_normalize(images)


cu.display_image(images[0], title='load frame 1')

#kernels = np.load(f"{image_folder}kernels.npy")

# Load the MATLAB file
#mat_data = scipy.io.loadmat(f"{image_folder}psf_1X_binning_data.mat")
mat_data = scipy.io.loadmat("{}psf_1X_binning_data.mat".format(image_folder))


# Extract the array
kernel = mat_data['psf_1X_binning_data']

# Convert to a NumPy array (if not already)
kernel = np.array(kernel)

# Display the array
#print("Loaded array from MATLAB:")
print("Loaded {} images.".format(len(images)))

print(kernel)


# Check the number of images loaded
#print(f"Loaded {len(images)} images.")
print("Loaded {} images.".format(len(images)))


# Print the shape of each loaded image
for idx, img in enumerate(images):
    #print(f"Shape of image {idx + 1}: {img.shape}")
    print("Shape of image {}: {}".format(idx + 1, img.shape))

    


# Define the dimensions of the problem
N = 3  # Number of dimensions, or the number of frames
mu = 0.1
rho = 0.7  # Step size or regularization parameter try 0.7 or 0.8 

v = 0
gamm = 0.9 


# Main iterative process
max_iterations = 10
tolerance = 1e-3

#image_size = 256                # Image size
#P = 10                          # Blur kernel with size (2P+1)x(2P+1)
#filter_std = 2.0                # spatial standard deviation of blur kernel
decimation_rate = 4             # Integer decimation rate
lambda_param = 0.8              # Seems to become numerically unstable for lambda_param < 0.5

#w = np.zeros(len(images), gt_image.shape)
w = np.zeros((len(images),) + gt_image.shape)

print(w.shape)

#ref_image = pnp.apply_G(gt_image, kernel, decimation_rate)

ref_image = gt_image # use high resolution image as GT image

## Apply Wiener filter for deblurring
#ref_image, _ = restoration.unsupervised_wiener(gt_image, kernel)

##ref_image, _, _ = cu.min_max_normalize(ref_image)
#ref_image = np.clip(ref_image, 0, 1)

print(ref_image.shape)
print(images[0].shape)

# Ensure the gt image is a NumPy array
ref_image = np.array(ref_image)
images = np.array(images)

cu.display_3images(gt_image, ref_image, images[0], title1 = 'high resolution image', title2='deblurred image', title3='loaded low resolution image')

# pad the kernal to the image size
kernel_padded = pnp.pad_kernel(kernel, images[0].shape)

kernels = np.expand_dims(kernel_padded, axis=0)
for i in range(len(images)):
    # updample the images
    shifted_image = images[i]
    shifted_image = resize(shifted_image, gt_image.shape, anti_aliasing=True)


    # Calculate the shift using phase cross-correlation
    calculated_shift, error, diffphase = phase_cross_correlation(shifted_image, ref_image)

    #print(f'Calculated offset (y, x): {calculated_shift}')
    #print(f'Error: {error}')
    #print(f'Diffphase: {diffphase}')

    print('Calculated offset (y, x): {}'.format(calculated_shift))
    print('Error: {}'.format(error))
    print('Diffphase: {}'.format(diffphase))

    # Apply the same shift to the PSF
    shifted_psf = fourier_shift(np.fft.fftn(kernel_padded), calculated_shift)
    shifted_psf = np.fft.ifftn(shifted_psf)
    shifted_psf = np.abs(shifted_psf)  # Take the magnitude to get the real part of the shifted PSF

    # Create a list of 100 copies of the sample array
    #array_list = [kernel for _ in range(len(images))]

    # Stack the arrays along a new axis (0 in this case)
    #kernels = np.stack(array_list, axis=0)
    kernels = np.concatenate((kernels,np.expand_dims(shifted_psf, axis=0)), axis=0)
    
    #print(f"Loaded {len(kernels)} kernels.")
    print("Loaded {} kernels.".format(len(kernels)))


# delete the first kernel 
kernels = kernels[1:]
#print(f"Loaded {len(kernels)} kernels.")
print("Loaded {} kernels.".format(len(kernels)))

#w = gt_images
#mu = 0.1

# MACE 

rmse_values = []
for iteration in tqdm(range(max_iterations)):
    # Step 1:
    x = F(w, images, kernels, decimation_rate, lambda_param)
    
    # Add a moving average kernel to smooth the image
    kernel_size = 2
    MA_kernel = np.ones((kernel_size,kernel_size), dtype=float) /(kernel_size**2)
    for i in range(1, x.shape[0]):
        x[i] = jax.scipy.signal.convolve(x[i], MA_kernel, mode="same") 


    # Step 2:
    z = G_mu(2 * x - w)
    
    # Step 3: add denoiser
    if DenoiserEnabled == 0: 
        # project data to (0,1) space
        normalized_z, min_val, max_val = cu.min_max_normalize(z)

        print('min and max:', min_val, max_val)

        """     # denoiser
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
        z = cu.min_max_denormalize(denoised_image, min_val, max_val)"""

        #denoiser_funtion = pnp.get_denoiser(method='GF')
        #denoiser_funtion = pnp.get_denoiser(method='BM3D')
        denoiser_funtion = pnp.get_denoiser(method='DPIR')
        

        denoised_image = np.zeros_like(z)
        denoised_image[0] = denoiser_funtion(z[0], 0.1)
        for i in range(1, z.shape[0]):
            denoised_image[i] = denoised_image[0] 

        z = denoised_image

    ## Apply Wiener filter for deblurring
    #denoised_image, _ = restoration.unsupervised_wiener(gt_image, kernel)

    ##ref_image, _, _ = cu.min_max_normalize(ref_image)
    #ref_image = np.clip(ref_image, 0, 1) 

    # Step 4


    v = gamm * v + 2 * rho * (1 - gamm) * (z - x)
    w_new = w + v


    # use simple rho
    #v = gamm * v + (1 - gamm) * rho
    #w_new = w + 2 * v * (z - x)



    #w_new = w + 2 * rho * (z - x)

    # Convergence check (stop if the update is small)
    if np.linalg.norm(w_new - w) < tolerance:
        #print(f"Converged after {iteration+1} iterations.")
        print("Converged after {} iterations.".format(iteration + 1))
        break
    
    w = w_new
    temp = z[0,:]
    #cu.display_image(temp, title='restored')
    rmse = pnp.mse(temp, gt_image)
    rmse_values.append(rmse)
    time.sleep(0.1)


# Return the result
x_star = z[0,:]
x_star, min_val, max_val = cu.min_max_normalize(x_star)


# Save to a binary file in NumPy `.npy` format
np.save('./data/rmse_values.npy', rmse_values)

# compute the mse
#rmse = pnp.mse(x_star, ref_image)
nrmse = pnp.nrmse(gt_image, x_star, kernel, decimation_rate)

#print(f"RMSE between the restored image and GT image is {rmse}")
print("RMSE between the restored image and GT image is {}".format(rmse))


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
