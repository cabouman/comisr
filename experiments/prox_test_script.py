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
    image_size = 512                # Image size
    P = 10                           # Blur kernel with size (2P+1)x(2P+1)
    filter_std = 2.0                # spatial standard deviation of blur kernel
    decimation_rate = 2             # Integer decimation rate
    lambda_param = 0.5              # Seems to become numerically unstable for lambda_param < 0.5

    # Load in the 1951 AF target
    file_path = 'data/USAF-1951.svg.png'
    image = cu.read_png(file_path)[:, :, 1]
    print(f'image.shape: {image.shape}')

    # Resize image so it is reasonable to work with
    image = cu.resize_image(image, new_shape=(image_size, image_size))
    gt_image = jnp.array(image)

    # Generate a gaussian kernel
    kernel = pnp.gen_gaussian_filter(P, filter_std)

    # Generate synthetic data
    measured_image = pnp.apply_G(gt_image, kernel, decimation_rate)

    print(f'gt_image.shape: {gt_image.shape}')
    print(f'measured_image.shape: {measured_image.shape}')


    # #########################
    # Sanity check: Initialize with ground truth and check that it doesn't change
    prox_image = gt_image

    NumIterations = 10
    for i in range(NumIterations):
        prox_image = pnp.proximal_map(prox_image, measured_image, kernel, decimation_rate, lambda_param )

    # Display ground truth and prox output
    cu.display_images(gt_image, prox_image, label1='Image 1', label2='Image 2')

    # #########################
    # Convergence check: Initialize with zeros and see if it converges to the ML estimate.
    prox_image = jnp.zeros(gt_image.shape)
    print(f'gt_image.shape: {gt_image.shape}')
    print(f'prox_image.shape: {prox_image.shape}')

    NumIterations = 10
    for i in range(NumIterations):
        prox_image = pnp.proximal_map(prox_image, measured_image, kernel, decimation_rate, lambda_param )

    # Display ground truth and prox output
    cu.display_images(measured_image, prox_image, label1='Measured Image', label2='{NumIterations} Iterations of Proximal Map}')
