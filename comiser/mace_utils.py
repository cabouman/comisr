# MACE 
import numpy as np
import pnp_utils as pnp

def F(w, measured_images, kernels, decimation_rate, lambda_param):
    for j in range(w.shape[0]):

        measured_image = measured_images[j,:]
        kernel = kernels[0,:]
        this_w = w[j,:]
        #cu.display_images(this_w, measured_image, title1='thisw', title2='measured')
        this_w = pnp.proximal_map_numerically_stable(this_w , measured_image, kernel, decimation_rate, lambda_param )
        #cu.display_image(this_w, title='proxi')

        w[j,:] = this_w

    return w  # Assuming A is some predefined matrix


def process_single_item(args):
    j, w_j, measured_image, kernel, decimation_rate, lambda_param = args
    # Apply the proximal map function
    w_j_updated = pnp.proximal_map_numerically_stable(w_j, measured_image, kernel, decimation_rate, lambda_param)
    return j, w_j_updated

from multiprocessing import Pool

def parallel_F(w, measured_images, kernels, decimation_rate, lambda_param):
    # Assume kernel is constant across all calls, adapt if necessary
    # kernel = kernels[0, :]

    # Prepare arguments for each process
    args = [(j, w[j, :], measured_images[j, :], kernels[j,:], decimation_rate, lambda_param) for j in range(w.shape[0])]

    # Number of processes
    num_processes = 4  # Adjust this based on your CPU

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_item, args)

    # Update w with the results
    for j, w_j_updated in results:
        w[j, :] = w_j_updated

    return w

# Dummy function G_mu, assuming some form of projection or transformation
def G_mu(x):
    new_array = np.ones_like(x)
    new_array = new_array * np.mean(x, axis=0)
    return new_array  # Example: Simple thresholding