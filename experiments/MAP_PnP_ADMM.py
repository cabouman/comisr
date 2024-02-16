import numpy as np
from scipy.optimize import minimize
import cv2

# Define the log likelihood function (assuming Gaussian likelihood)
def log_likelihood(theta, data):
    mu, sigma = theta
    return -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + (data - mu)**2 / sigma**2)

# Define the log prior function (assuming Gaussian prior)
def log_prior(theta):
    mu, sigma = theta
    prior_mu = 0  # Mean of the prior
    prior_sigma = 1  # Standard deviation of the prior
    return -0.5 * np.sum(np.log(2 * np.pi * prior_sigma**2) + (theta - prior_mu)**2 / prior_sigma**2)

# Define the log posterior function
def log_posterior(theta, data):
    return log_likelihood(theta, data) + log_prior(theta)


# Define the typical MAP optimization problem
# where f(x) is th negative log likelihood, g(x) is the negative log prior probability
# x_hat = argmin_x {f(x)+g(x)}
def map_estimation(log_likelihood, log_prior, initial_guess, data):
    """
    Perform Maximum A Posteriori (MAP) estimation.

    Parameters:
        likelihood (callable): Likelihood function that takes parameters and data.
        prior (callable): Prior distribution function that takes parameters.
        initial_guess (numpy.ndarray): Initial guess for the parameters.
        data: Observed data.

    Returns:
        numpy.ndarray: Estimated parameters that maximize the posterior.
    """

    # Use scipy.optimize.minimize to find the parameters that maximize the negative log posterior
    #result = minimize(lambda theta: negative_log_posterior, initial_guess)
    result = minimize(lambda theta: -log_posterior(theta, data), initial_guess)


    return result.x

def ADMM_SR(y, rho, max_iter):
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
    m,n = y.shape
    print("Data shape:", y.shape)


    # Initialize variables
    x = np.zeros(shape=(m,n), dtype=y.dtype)
    v = np.zeros(shape=(m,n), dtype=y.dtype)
    u = np.zeros(shape=(m,n), dtype=y.dtype)

    #rows_in,cols_in = y.shape
    #rows  = np.dot(rows_in, K)
    #cols  = np.dot(cols_in,K)
    #N     = np.dot(rows,cols)

    # G,Gt    = defGGt(h,K);  
    # GGt       = constructGGt(h,K,rows,cols);
    # Gty       = Gt(y);
    # v         = imresize(y,K);
    # x         = v;
    # u         = zeros(size(v));
    # residual  = inf;


    # ADMM iterations
    for k in range(max_iter):
        # Update x
        x = map_estimation(log_likelihood, log_prior, initial_guess, y)

        #xtilde = v-u
        #rhs = Gty + np.dot(rho, xtilde)
        #x = (rhs - Gt(ifft2(fft2(G(rhs))./(GGt + rho))))/rho;

        # Update v
        v = x + u
        #v = np.clip(v, 0, 255)
        # vtilde = x+u
        # vtilde = proj(vtilde)
        # sigma  = sqrt(lambda/rho)
        # v      = denoise(vtilde,sigma);

        # Update u
        u = u + x - v

    return x

