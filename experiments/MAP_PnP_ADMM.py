import numpy as np
from scipy.optimize import minimize
import cv2
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
# from skimage.restoration import denoise_bm3d
import approximal_map

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
            g[i + int(np.floor(L / 2)), j + int(np.floor(L / 2))] = hth[int(yc + K * i), int(xc + K * j)]
    
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


# """ # Define the log likelihood function (assuming Gaussian likelihood)
# def log_likelihood(theta, data):
#     mu, sigma = theta
#     return -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + (data - mu)**2 / sigma**2)

# # Define the log prior function (assuming Gaussian prior)
# def log_prior(theta):
#     mu, sigma = theta
#     prior_mu = 0  # Mean of the prior
#     prior_sigma = 1  # Standard deviation of the prior
#     return -0.5 * np.sum(np.log(2 * np.pi * prior_sigma**2) + (theta - prior_mu)**2 / prior_sigma**2)

# # Define the log posterior function
# def log_posterior(theta, data):
#     return log_likelihood(theta, data) + log_prior(theta)


# # Define the typical MAP optimization problem
# # where f(x) is th negative log likelihood, g(x) is the negative log prior probability
# # x_hat = argmin_x {f(x)+g(x)}
# def map_estimation(log_likelihood, log_prior, initial_guess, data):
#     """
#     Perform Maximum A Posteriori (MAP) estimation.

#     Parameters:
#         likelihood (callable): Likelihood function that takes parameters and data.
#         prior (callable): Prior distribution function that takes parameters.
#         initial_guess (numpy.ndarray): Initial guess for the parameters.
#         data: Observed data.

#     Returns:
#         numpy.ndarray: Estimated parameters that maximize the posterior.
#     """

#     # Use scipy.optimize.minimize to find the parameters that maximize the negative log posterior
#     #result = minimize(lambda theta: negative_log_posterior, initial_guess)
#     result = minimize(lambda theta: -log_posterior(theta, data), initial_guess)


#     return result.x

# def inverse_updateX(y,rho,xtilde,Gty):
#     # Need more details here 
#     rhs = Gty + np.dot(rho, xtilde)
#     x = rhs """



# def denoiser_BM3D(x,u,lamda, rho):
#     vtilde = x + u
#     vtilde = proj(vtilde)
#     sigma  = np.sqrt(lamda/rho)
#     # Denoise the image using BM3D
#     #image_denoised = denoise_bm3d(image_noisy, sigma_psd=0.2, stage_arg=[{'refilter': True}])
#     v = denoise_bm3d(vtilde,sigma_psd=sigma, stage_arg=[{'refilter': True}])
#     return v 


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
    
    max_iter = 3
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

    G,Gt = defGGt(h,K)
    GGt = construct_GGt(h,K,rows,cols)
    Gty = Gt(y)
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

        # Inverse step
        xtilde = v-u
        #rhs = Gty + np.dot(rho, xtilde)
        #x = (rhs - Gt(np.fft.ifft2(np.fft.fft2(G(rhs))/(GGt + rho))))/rho

        x = approximal_map.approximal_map_F(h,K,rho,y,xtilde)


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


