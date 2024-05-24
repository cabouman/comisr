from skimage import restoration

# Denoiser using  Linear Filter 
def NLM(image, sigma):
    # Use Non-Local Means denoising without the 'multichannel' parameter
    denoised = restoration.denoise_nl_means(image,
                                            h=1.15 * sigma,
                                            fast_mode=True,
                                            patch_size=7,
                                            patch_distance=11)
    return denoised

