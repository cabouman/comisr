
import numpy as np
import matplotlib.pyplot as plt
import comiser.pnp_utils as pnp 



def admm_with_proximal(x, y, kernel, decimation_rate, lambda_param, rho, mu, max_iter=1000, tol=1e-4):
    """
    ADMM for solving:
    minimize_x f(x)+ rho |x-(v-u)|^2 using the proximal map.
    f(x) = |y-Gx|^2
    
    Parameters:
    x: prox_input_image
    y: measured_image
    rho : float
        Penalty parameter for the ADMM.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for the stopping criterion.
    """
    M, N = y.shape

    m = M * decimation_rate
    n = N * decimation_rate

    u = np.zeros((m,n))
    v = np.zeros((m,n))
    

    for iteration in range(max_iter):

        # x-update (solving the quadratic subproblem)
        x_tilda = v - u 
        x = pnp.proximal_map_numerically_stable(x_tilda, y, kernel, decimation_rate, lambda_param)

        # z-update (applying the proximal map for the L1 norm)
        z_tilde = x + u
        z_old = z_tilde.copy()
        z = soft_thresholding(z_tilde, mu / rho)
        
        # u-update (dual variable update)
        u += x_tilda - z
        
        # Convergence check
        if np.linalg.norm(x - z_old) < tol:
            print(f"Convergence reached after {iteration + 1} iterations.")
            break

    return x


def soft_thresholding(v, lambda_):
    """ Soft thresholding operator for L1 regularization. """
    return np.sign(v) * np.maximum(np.abs(v) - lambda_, 0)