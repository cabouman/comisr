from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

# add parent path to import functions in the comiser folder 
import sys
sys.path.append('..')  
import comiser.utils as cu
import comiser.pnp_utils as pnp
import comiser.img_utils as cimgu

#mport comiser.admm_utils as admm

# Use double precision floats
jax.config.update("jax_enable_x64", True)


"""
This is a script to test the proximal map functions  
"""
if __name__ == "__main__":
    image_size = 256                # Image size
    P = 2                           # Blur kernel with size (2P+1)x(2P+1)
    filter_std = 2.0                # spatial standard deviation of blur kernel
    decimation_rate = 2             # Integer decimation rate
    lambda_param = 0.3       # Seems to become numerically unstable for lambda_param < 0.5

    # Load in the 1951 AF target
    file_path = 'data/USAF-1951.svg.png'
    image = cu.read_png(file_path)[:, :, 1]
    print(f'original image.shape: {image.shape}')

    # Resize image so it is reasonable to work with
    image = cu.resize_image(image, new_shape=(image_size, image_size))
    gt_image = jnp.array(image)
    print(f'ground truth image shape: {gt_image.shape}')

    # Generate a gaussian kernel
    kernel = pnp.gen_gaussian_filter(P, filter_std)

    # Generate synthetic image
    measured_image = pnp.apply_G(gt_image, kernel, decimation_rate)

    # #####################
    # Compute Wiener filter psf
    desired_shape = pnp.get_odd_filter_shape(gt_image.shape, K=4)
    wiener_psf = pnp.gen_wiener_filter_psf(kernel, decimation_rate, lambda_param, desired_shape)

    # Display results
    cu.display_image(wiener_psf, title=f'Wiener PSF lambda = {lambda_param} shape = {desired_shape[0]}')
    # #####################


    # #########################
    # Sanity check: Initialize with ground truth and check that it doesn't change
    prox_image = gt_image

    NumIterations = 1
    for i in range(NumIterations):
        prox_image = pnp.proximal_map_numerically_stable(prox_image, measured_image, kernel, decimation_rate, lambda_param )

    # Display ground truth and prox output
    cu.display_images(gt_image, prox_image, title1='Ground Truth', title2='Prox Output Image')


    # #########################
    # Convergence check: Initialize with zeros and see if it converges to the ML estimate.
    prox_image = jnp.zeros(gt_image.shape)
    # prox_image_start = prox_image

    prox_image1 = prox_image

    NumIterations = 20
    
    # for i in range(NumIterations):
    #     prox_image1 = pnp.proximal_map_numerically_stable(prox_image1, measured_image, kernel, decimation_rate, lambda_param)

    #     # Test that iterated prox is converging to the correct solution
    #     nrmse1 = pnp.get_nrmse_convergence_error(measured_image, prox_image1, kernel, decimation_rate)
    #     print(f'RMSE 1 = {nrmse1}')

    # # Display ground truth and prox output
    # cu.display_3images(gt_image, measured_image, prox_image1, title1 = 'Ground Truth', title2='Measured Image', title3=f'{NumIterations} Iterations of Proximal Map')


    # Apply the ADMM
    restored_image = jnp.zeros(gt_image.shape)
    # denoiser_kernel = pnp.gen_gaussian_filter(2*P, filter_std/4)
    sigma_denoiser = 0.1
    denoiser_method = "BM3D"
    restored_image = pnp.admm_with_proximal(restored_image, measured_image, kernel, decimation_rate, lambda_param, denoiser_method, sigma_denoiser, max_iter = NumIterations, tol=1e-5)
    cu.display_3images(gt_image, measured_image, restored_image, title1='Ground Truth', title2 = 'Measured Image', title3=f'{NumIterations} Iterations of ADMM')
