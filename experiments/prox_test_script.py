from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import comiser.utils as cu
import comiser.pnp_utils as pnp


"""
This is a script to test the proximal map functions  
"""
if __name__ == "__main__":
    image_size = 256                # Image size
    P = 2                           # Blur kernel with size (2P+1)x(2P+1)
    filter_std = 2.0                # spatial standard deviation of blur kernel
    decimation_rate = 2             # Integer decimation rate
    lambda_param = 0.2              # Seems to become numerically unstable for lambda_param < 0.5

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
    cu.display_image(wiener_psf, title='Wiener PSF')
    # #####################


    # #########################
    # Sanity check: Initialize with ground truth and check that it doesn't change
    prox_image = gt_image

    NumIterations = 10
    for i in range(NumIterations):
        prox_image = pnp.proximal_map_numerically_stable(prox_image, measured_image, kernel, decimation_rate, lambda_param )

    # Display ground truth and prox output
    cu.display_images(gt_image, prox_image, title1='Ground Truth', title2='Prox Output Image')


    # #########################
    # Convergence check: Initialize with zeros and see if it converges to the ML estimate.
    prox_image = jnp.zeros(gt_image.shape)
    prox_image_start = prox_image

    NumIterations = 100
    for i in range(NumIterations):
        prox_image = pnp.proximal_map_numerically_stable(prox_image, measured_image, kernel, decimation_rate, lambda_param )

    # Test that iterated prox has reached the correct solution
    error_image = measured_image - pnp.apply_G(prox_image, kernel, decimation_rate)
    nrmse = jnp.sqrt(jnp.sum(error_image ** 2)/jnp.sum(measured_image ** 2))
    print(f'RMSE = {nrmse}')

    # Display ground truth and prox output
    cu.display_images(measured_image, prox_image, title1='Measured Image', title2=f'{NumIterations} Iterations of Proximal Map')
