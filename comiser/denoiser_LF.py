from skimage import restoration
from skimage import filters

# Denoiser using  Linear Filter 

def GaussianFilter(image, sigma):
    # Only set the sigma value; the skimage function will select the optimized filter size that is large enough to effectively 
    # apply the Gaussian function.
    
    filtered_image = filters.gaussian(image, sigma=sigma)
    return filtered_image


def NLM(image, sigma):
    # Use Non-Local Means denoising without the 'multichannel' parameter
    denoised = restoration.denoise_nl_means(image,
                                            h=1.15 * sigma,
                                            fast_mode=True,
                                            patch_size=7,
                                            patch_distance=11)
    return denoised