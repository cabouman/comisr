import numpy as np
from scipy.optimize import minimize
import cv2
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
# from skimage.restoration import denoise_bm3d
import proximal_map as proximal_map


def proj(x, bound=None):
    """
    Project a vector x onto the specified interval defined by bound.

    Args:
    - x: input vector
    - bound: 2-element list or tuple specifying the interval [lower_bound, upper_bound].
             Default is [0, 1].

    Returns:
    - out: projected vector

    Example:
    out = proj(x, [1, 3]) projects a vector x onto the interval [1, 3]
    by setting x values greater than 3 to 3, and x values less than 1 to 1.
    """
    if bound is None:
        bound = [0, 1]
    out = np.minimum(np.maximum(x, bound[0]), bound[1])
    return out


def PlugPlayADMM_super(y, h, K, lam):

    """
    Alternating Direction Method of Multipliers (ADMM) for solving least squares problem.

    Parameters:
        A (numpy.ndarray): Design matrix.
        b (numpy.ndarray): Target vector.
        rho (float): Penalty parameter.
        max_iter (int): Maximum number of iterations.

    Returns:
        numpy.ndarray: Solution vector.
    """
    
    max_iter = 10
    rho = 1
    gamma = 1

    m,n = y.shape
    print("Data shape:", y.shape)

    rows_in,cols_in = y.shape
    rows  = np.dot(rows_in, K)
    cols  = np.dot(cols_in,K)
    N     = np.dot(rows,cols)

    # Initialize variables
    x = np.zeros(shape=(rows,cols), dtype=y.dtype)
    v = np.zeros(shape=(rows,cols), dtype=y.dtype)
    u = np.zeros(shape=(rows,cols), dtype=y.dtype)

    #G,Gt = defGGt(h,K)
    #GGt = construct_GGt(h,K,rows,cols)
    #Gty = Gt(y)
    v   = cv2.resize(y, (rows_in*K, cols_in*K))
    x   = v
    residual  = np.inf


    # ADMM iterations
    for k in range(max_iter):
        print(k)
        # Update x
        #x = map_estimation(log_likelihood, log_prior, initial_guess, y)

        # store x, v, u from previous iteration for psnr residual calculation
        x_old = x
        v_old = v
        u_old = u

        # Inverse step ( The proximal map solution of the forward model)
        xtilde = v-u
        #rhs = Gty + np.dot(rho, xtilde)
        #x = (rhs - Gt(np.fft.ifft2(np.fft.fft2(G(rhs))/(GGt + rho))))/rho
        x = proximal_map.proximal_map_F(xtilde,h,K,rho,y)

        # Update v
        v = x + u
        v = np.clip(v, 0, 255)
        vtilde = x + u
        vtilde = proj(vtilde)
        sigma = np.sqrt(lam/rho)
        # Denoise the grayscale image
        # denoised_image = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        #v = cv2.fastNlMeansDenoising(vtilde, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # Convert the image to float32
        vtilde = vtilde.astype(np.float32)

        # Denoise the grayscale image using Non-Local Means Denoising
        # v = cv2.fastNlMeansDenoising(vtilde, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # v = denoise_bm3d(vtilde,sigma)
        
        # Apply Gaussian blur for denoising
        #denoised_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
        v = cv2.GaussianBlur(vtilde, (5, 5), 0)

        # Update u
        u = u + x - v

        # update rho  % Yufang: disable this step
        rho=rho*gamma

        # Calculate NMSE of residule
        # Calculate NMSE of residual for x
        residualx = np.sum(np.square(x - x_old))
        residualx = residualx / np.sum(np.square(x))

        # Calculate NMSE of residual for v
        residualv = np.sum(np.square(v - v_old))
        residualv = residualv / np.sum(np.square(v))

        # Calculate NMSE of residual for u
        residualu = np.sum(np.square(u - u_old))
        residualu = residualu / np.sum(np.square(u))

        residual = residualx + residualv + residualu
        print(residual)

    return x


