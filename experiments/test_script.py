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

    # Set user-defined parameters
    P = 2                           # Blur kernal with size (2P+1)x(2P+1)
    filter_std = 2.0                # spatial standard deviation of blur kernel
    image_size = 512                # in pixels
    sigma_x = 1.0                   # Set proximal map parameter
    sigma_y = 1.0                   # Set noise variance parameter
    verbose = 0                     # Set debugging level: 0 - fast; 1 - test more stuff

    # Load in the 1951 AF target
    file_path = 'data/USAF-1951.svg.png'
    image = cu.read_png(file_path)[:, :, 1]
    print(f'image.shape: {image.shape}')

    new_shape = (image_size, image_size)
    image = cu.resize_float_array_using_pil(image, new_shape)
    image = jnp.array(image)

    # Generate a gaussian kernel
    kernel = pnp.gaussian_filter(P, filter_std)

    # Filter image
    image = pnp.filter_2D_jax(image, kernel)

    cu.display_image(image)


