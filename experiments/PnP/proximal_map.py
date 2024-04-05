import numpy as np
from scipy.optimize import minimize
import cv2
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
# from skimage.restoration import denoise_bm3d


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



# The approximal map implementation using Fourier transform:
# input parameters: h, K, 𝜌
# input data: 𝑦, 𝑥
# output: closed form solution of the approximal map using Fourier transformation 

def proximal_map_F(h,K,rho,y,xtilde):
    # y: input image
    # h: psf or bulerring filter
    # K: down/up samping factor
    # rho: 

    rows_in,cols_in = y.shape
    rows  = np.dot(rows_in, K)
    cols  = np.dot(cols_in,K)

    G,Gt = defGGt(h,K)
    GGt = construct_GGt(h,K,rows,cols)
    Gty = Gt(y)
    #v   = cv2.resize(y, (rows_in*K, cols_in*K))
    #x   = v

    # Note: xtilde = v-u in the PnP ADMM
    # solve the closed form:
    rhs = Gty + np.dot(rho, xtilde)
    x = (rhs - Gt(np.fft.ifft2(np.fft.fft2(G(rhs))/(GGt + rho))))/rho

    return x
