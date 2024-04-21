import types
import numpy as np
import yaml
import warnings
import gc
import jax
import jax.numpy as jnp


def proximal_map(prox_input, measured_image, kernel, decimation_rate, lambda_param ):
    """
    Compute the proximal map funtion
    Returns:
    """
    # Compute temporary image denoted by b in notes
    epsilon_image = measured_image - apply_G(prox_input, kernel, decimation_rate)
    b_image = apply_Gt(epsilon_image, kernel, decimation_rate)
    c_image = apply_G(b_image, kernel, decimation_rate)

    # Generate the special filter kernel that will be needed.
    htilde0 = get_htilde0(kernel, decimation_rate)

    # Pad and shift the kernel so its FFT will be real valued.
    htilde0_padded = pad_and_shift_kernel(htilde0, measured_image.shape)

    # Compute the transfer functino associated
    transfer_function = 1.0 /( np.fft.fft2(htilde0_padded) + (lambda_param ** 2) )
    real_transfer_function = transfer_function.real

    # The FFT should be real valued, so check that this is true.
    error = jnp.max(jnp.abs(transfer_function.imag))
    if error > 1e-6:
        warnings.warn("The error = max(abs(imaginary part)) is large.", UserWarning)

    # Calculate Wiener filtered signal
    d_image = (np.fft.ifft2(real_transfer_function * np.fft.fft2(c_image))).real

    # Compute the output change
    delta_output = (lambda_param ** 2)*(b_image - apply_Gt(d_image, kernel, decimation_rate))
    print(f'delta_output max absolute value: {jnp.max(jnp.abs(delta_output))}')

    # Compute the prox output
    prox_output_image = delta_output + prox_input
    return prox_output_image


def filter_2D_jax(image, kernel):
    """
    Applies a 2D filter to an image using JAX.

    Parameters
    ----------
    image : jnp.ndarray
        The input grayscale image to which the filter is to be applied.
    kernel : jnp.ndarray
        The kernel to be applied on the image.

    Returns
    -------
    jnp.ndarray
        The filtered image after applying the filter.
    """
    result = jax.scipy.signal.convolve(image, kernel, mode='same')
    return result

def get_htilde0(kernel, decimation_rate):
    """
    Returns the filter htilde_0 which is formed by taking the autocorrelation of kernel with itself and subsampling
    along each axis by a factor of decimation_rate.
    Args:
        kernel (jnp.ndarray): 2D kernel
        decimation_rate (int): Integer decimation rate.

    Returns:
        jnp.ndarray
    """
    htilde_0 = filter_2D_jax(kernel, flip_array(kernel))[::decimation_rate, ::decimation_rate]

    return htilde_0

def flip_array(arr):
    """
    Flip a 2D array along both axes.

    Args:
    image (np.ndarray): Input 2D array.

    Returns:
    np.ndarray: The array flipped along both vertical and horizontal axes.
    """
    # Flip the array vertically and horizontally
    flipped_arr = arr[::-1, ::-1]
    return flipped_arr


def upsample_image_jax(image, upsample_rate):
    """
    Upsample a 2D JAX array by inserting zeros to increase its size by a factor of upsample_rate in each direction.

    Args:
    image (jnp.ndarray): The input 2D JAX array.
    upsample_rate (int): The upsampling factor, specifying how much to enlarge the array.

    Returns:
    jnp.ndarray: An upsampled 2D JAX array with size increased by a factor of upsample_rate.
    """
    # Determine the new shape
    rows, cols = image.shape
    upsampled_shape = (rows * upsample_rate, cols * upsample_rate)
    # Create a new JAX array of zeros with the upsampled shape
    upsampled_image = jnp.zeros(upsampled_shape, dtype=image.dtype)

    # Insert the original elements at the correct positions
    upsampled_image = upsampled_image.at[::upsample_rate, ::upsample_rate].set(image)

    return upsampled_image



def gen_gaussian_filter(P, filter_std):
    """
    Generates a Gaussian filter of size (2P+1)x(2P+1) with standard deviation filter_std.

    Args:
    P (int): Determines the size of the filter as (2P+1)x(2P+1).
    filter_std (float): Standard deviation of the Gaussian blur.

    Returns:
    jnp.ndarray: A (2P+1)x(2P+1) Gaussian filter.
    """
    size = 2 * P + 1  # Compute the size of the filter
    kernel = np.zeros((size, size))  # Initialize the kernel array

    for i in range(size):
        for j in range(size):
            x = i - P
            y = j - P
            # Calculate the Gaussian function
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * filter_std**2))

    # Normalize the kernel so that the sum of all elements is 1
    kernel /= 2 * np.pi * filter_std**2
    kernel /= np.sum(kernel)
    kernel = jnp.array(kernel)

    return kernel


def apply_G(image, kernel, decimation_rate):
    """
    This function filters with the kernel and then subsamples by decimation_rate.
    Args:
        image:
        kernel:
        decimation_rate:

    Returns:

    """
    image = filter_2D_jax(image, kernel)[::decimation_rate, ::decimation_rate]
    return image


def apply_Gt(image, kernel, decimation_rate):
    """
    This function upsamples by decimation_rate and then filters with time-reversed kernel.
    Args:
        image:
        kernel:
        decimation_rate:

    Returns:
        image:
    """
    # Upsample image
    upsampled_image = upsample_image_jax(image, decimation_rate)

    # Filter with flipped kernel
    flipped_kernel = flip_array(kernel)
    output_image = filter_2D_jax(upsampled_image, flipped_kernel)
    return output_image

def pad_and_shift_kernel(kernel, image_shape):
    """
    Pad a small MxM kernel to the size of a given image and circularly shift
    it so that the center of the kernel moves to the (0,0) position.

    Args:
    kernel (np.ndarray): The input MxM kernel (M should be odd).
    image_shape (tuple): The shape of the large 2D image (height, width).

    Returns:
    np.ndarray: A padded and shifted kernel of the same size as the image.
    """
    M = kernel.shape[0]  # Assume kernel is square and M = 2P + 1
    P = (M - 1) // 2      # Calculate P based on M

    # Create an array of zeros with the same shape as the image
    padded_kernel = np.zeros(image_shape)

    # Calculate the indices where the kernel should be placed
    center = P  # Since kernel is MxM and M=2P+1, center is at P

    # Place the kernel into the padded array (centered in the middle)
    padded_kernel[:M, :M] = kernel

    # Perform circular shift so that the center of the kernel goes to (0,0)
    padded_kernel = np.roll(padded_kernel, -center, axis=0)
    padded_kernel = np.roll(padded_kernel, -center, axis=1)

    return padded_kernel

