import numpy as np
import jax.numpy as jnp


# add parent path to import functions in the comiser folder 
import sys
import os
sys.path.append('..')  
sys.path.append(os.path.abspath('..'))
#import comiser.pnp_utils as pnp
import comiser.utils as cu
#import comiser.img_utils as cimgu
import comiser.mace_utils as mace


""" # Example function F, which could be a linear transformation or any application-specific function

def F(w, measured_images, kernels, decimation_rate, lambda_param):
    for j in range(w.shape[0]):

        measured_image = measured_images[j,:]
        kernel = kernels[0,:]
        this_w = w[j,:]
        #cu.display_images(this_w, measured_image, title1='thisw', title2='measured')
        this_w = pnp.proximal_map_numerically_stable(this_w , measured_image, kernel, decimation_rate, lambda_param )
        #cu.display_image(this_w, title='proxi')

        # for i in range(10):
        #     measured_image = measured_images[j,:]
        #     kernel = kernels[j,:]
        #     #decimation_rate = 2
        #     #lambda_param = 2
        #     this_w = w[j,:]

        #     this_w = pnp.proximal_map_numerically_stable(this_w , measured_image, kernel, decimation_rate, lambda_param )
        # # prox_image, measured_image, kernel, decimation_rate, lambda_param
        w[j,:] = this_w

    return w  # Assuming A is some predefined matrix


def process_single_item(args):
    j, w_j, measured_image, kernel, decimation_rate, lambda_param = args
    # Apply the proximal map function
    w_j_updated = pnp.proximal_map_numerically_stable(w_j, measured_image, kernel, decimation_rate, lambda_param)
    return j, w_j_updated

from multiprocessing import Pool

def parallel_F(w, measured_images, kernels, decimation_rate, lambda_param):
    # Assume kernel is constant across all calls, adapt if necessary
    kernel = kernels[0, :]

    # Prepare arguments for each process
    args = [(j, w[j, :], measured_images[j, :], kernel, decimation_rate, lambda_param) for j in range(w.shape[0])]

    # Number of processes
    num_processes = 5  # Adjust this based on your CPU

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_item, args)

    # Update w with the results
    for j, w_j_updated in results:
        w[j, :] = w_j_updated

    return w

# Dummy function G_mu, assuming some form of projection or transformation
def G_mu(x):
    new_array = np.ones_like(x)
    new_array = new_array * np.mean(x, axis=0)
    return new_array  # Example: Simple thresholding """

# # Example of a proximal operator function G_mu, implementing soft thresholding
# def G_mu(x, mu):
#     # Soft thresholding as an example of a proximal map
#     return np.sign(x) * np.maximum(np.abs(x) - mu, 0)
if __name__ == '__main__':

    # Define the dimensions of the problem
    N = 3  # Number of dimensions, or the number of frames
    mu = 0.1
    rho = 0.5  # Step size or regularization parameterls

    # Main iterative process
    max_iterations = 10
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

    # Generate a gaussian kernel
    kernel = pnp.gen_gaussian_filter(P, 2.0)
    measured_image = pnp.apply_G(gt_image, kernel, decimation_rate)

    kernels = np.expand_dims(kernel, axis=0)
    measured_images = np.expand_dims(measured_image, axis=0)
    gt_image_3dim = np.expand_dims(gt_image, axis=0)
    gt_images = np.expand_dims(gt_image, axis=0)

    for fn in range(9):
        shiftx = np.random.rand() * 0.5
        shifty = np.random.rand() * 0.5
        kernel_shift = cimgu.fft_subpixel_shift(kernel, shiftx, shifty)
        measured_image_shift = pnp.apply_G(gt_image, kernel_shift, decimation_rate)

        kernel_shift = np.expand_dims(kernel_shift, axis=0)
        measured_image_shift = np.expand_dims(measured_image_shift, axis=0)
        # stack 
        kernels = np.concatenate((kernels, kernel_shift), axis=0)
        measured_images = np.concatenate((measured_images, measured_image_shift), axis=0)
        gt_images = np.concatenate((gt_images, gt_image_3dim), axis=0)

    print("Shape of combined image array:", measured_images.shape)
    print("Shape of combined kernel array:", kernels.shape)


# Initialize x and w
#x0 = np.random.rand(N)  # Random initial vector
#w = np.tile(x0, (N, 1))  # Initialize w as a matrix with each row as x

    w = np.zeros(gt_images.shape)
#w = gt_images
#mu = 0.1

    # MACE 
    for iteration in range(max_iterations):
        #x = F(w, measured_images, kernels, decimation_rate, lambda_param)

        x = mace.parallel_F(w, measured_images, kernels, decimation_rate, lambda_param)

        z = mace.G_mu(2 * x - w)
        w_new = w + 2 * rho * (z - x)

        # Convergence check (stop if the update is small)
        if np.linalg.norm(w_new - w) < tolerance:
            print(f"Converged after {iteration+1} iterations.")
            break
    
        w = w_new
        temp = z[0,:]
        #cu.display_image(temp, title='restored')


    # Return the result
    x_star = z[0,:]

    # compute the mse
    rmse = pnp.mse(x_star, gt_image)
    print(f"RMSE between the restored image and GT image is {rmse}")

    restored_image = cu.convert_jax_to_image(x_star)
    restored_image.save('./data/restored_image_mace.png')
