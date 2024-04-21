import types
import numpy as np
import yaml
import warnings
import gc
import jax
import jax.numpy as jnp


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


def gaussian_filter(P, filter_std):
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


def construct_GGt(h, K, rows, cols):
    # Eigen-decomposition for super-resolution
    
    hth = convolve2d(h, np.rot90(h, 2), mode='full')
    
    yc = np.ceil(hth.shape[0] / 2)  # mark the center coordinate
    xc = np.ceil(hth.shape[1] / 2)
    
    L = np.floor(hth.shape[0] / K)  # width of the new filter
                                     # = (1/k) with of the original filter
    
    g = np.zeros((int(L), int(L)))  # initialize new filter
    for i in range(-int(np.floor(L / 2)), int(np.floor(L / 2)) + 1):
        for j in range(-int(np.floor(L / 2)), int(np.floor(L / 2)) + 1):
            g[i + int(np.floor(L / 2))-1, j + int(np.floor(L / 2))-1] = hth[int(yc + K * i-1), int(xc + K * j-1)]
    
    GGt = np.abs(np.fft.fft2(g, (int(rows / K), int(cols / K))))
    
    return GGt



def defGGt(h, K):
    """
    Operators for super-resolution
    """
    def G(x):
        return fdown(x, h, K)
    
    def Gt(x):
        return upf(x, h, K)
    
    return G, Gt

def fdown(x, h, K):
    tmp = fftconvolve(x, h, mode='same')
    y = downsample2(tmp, K)
    return y

def upf(x, h, K):
    tmp = upsample2(x, K)
    y = fftconvolve(tmp, h, mode='same')
    return y

def downsample2(x, K):
    return x[::K, ::K]

def upsample2(x, K):
    return np.kron(x, np.ones((K, K)))



# The approximal map implementation using Fourier transform:
# input parameters: filter_psf, subsampling_rate, 𝜌
# input data: 𝑦, 𝑥
# output: closed form solution of the approximal map using Fourier transformation 

def proximal_map_F(input_image, filter_psf, subsampling_rate, lambda_param, measured_image):
    """
    Computes the proximal map F(z) with parameters h, k, lambda_param, y
    
    Args:
        input_image (np.array): 2D input image
        filter_psf (np.array): 2D filter with odd width and height 
        subsampling_rate (integer): 
        lambda_param (scalar): lambda_param = (sigma_y^2/sigma_p^2)
        measured_image (np.array): 2D measured image:

    Returns:

    """
    rows_in, cols_in = measured_image.shape
    rows  = np.dot(rows_in, subsampling_rate)
    cols  = np.dot(cols_in, subsampling_rate)

    G,Gt = defGGt(filter_psf, subsampling_rate)
    GGt = construct_GGt(filter_psf, subsampling_rate, rows, cols)
    Gty = Gt(measured_image)
    #v   = cv2.resize(measured_image, (rows_in*subsampling_rate, cols_in*subsampling_rate))
    #x   = v

    # Note: input_image = v-u in the PnP ADMM
    # solve the closed form:
    rhs = Gty + np.dot(lambda_param, input_image)
    x = (rhs - Gt(np.fft.ifft2(np.fft.fft2(G(rhs)) / (GGt + lambda_param)))) / lambda_param

    return x
