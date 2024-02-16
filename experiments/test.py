import numpy as np
import MAP_PnP_ADMM
import cv2

# # Example usage:
# # Define likelihood function (Gaussian likelihood)
# def likelihood(params, data):
#     mu, sigma = params
#     return np.prod(1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((data - mu) / sigma)**2))

# # Define prior distribution (Gaussian prior)
# def prior(params):
#     mu, sigma = params
#     return np.exp(-0.5 * (mu**2 + sigma**2))

# # Define the negative log posterior function
# def negative_log_posterior(params):
#     return -np.log(likelihood(params, data)) - np.log(prior(params))

# Generate some example data
np.random.seed(0)
data = np.random.normal(loc=5, scale=2, size=1000)

# Initial guess for parameters
initial_guess = [0, 1]

# Perform MAP estimation
estimated_params = MAP_PnP_ADMM.map_estimation(MAP_PnP_ADMM.log_likelihood, MAP_PnP_ADMM.log_prior, initial_guess, data)
print("MAP estimation parameters (mu, sigma):", estimated_params)

# # Solve least squares problem using ADMM
# img = cv2.imread('./input/my_video_frame30.png', 0) 

# rho = 1.0
# max_iter = 100
# solution = MAP_PnP_ADMM.ADMM_SR(img, rho, max_iter)

# print("Solution:", solution)