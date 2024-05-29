import bm3d
from skimage import img_as_float

# Denoise the image using BM4D
def my_BM3D(image_noisy, sigma=0.1):
    image_noisy = img_as_float(image_noisy)
    image_denoised = bm3d.bm3d(image_noisy, sigma)
    return image_denoised