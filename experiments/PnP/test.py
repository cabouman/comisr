import numpy as np
import cv2
import MAP_PnP_ADMM
import approximal_map
K = 4


""" # Prepare the blurred data 
# Read the high-resolution image
z = cv2.imread('../../data/high_resolution_frame.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('OrigImage.png', z)

# Convert image to double precision
z = z.astype(np.float64) / 255.0

# Create a Gaussian filter
h = cv2.GaussianBlur(np.zeros_like(z), (9, 9), sigmaX=0.5)

# Downsample the image
y = cv2.resize(z, (z.shape[1]//K, z.shape[0]//K))
cv2.imwrite('DownImage.png', y*255)

# Add Gaussian noise
noise_level = 10 / 255.0
np.random.seed(0)  # Set random seed for reproducibility
noise = noise_level * np.random.randn(*y.shape)
y += noise 

# Write the denoised image to a file
cv2.imwrite('NoisyImage.png', y*255)
"""

y = cv2.imread('NoisyImage.png', cv2.IMREAD_GRAYSCALE)

# Convert image to double precision
y = y.astype(np.float64) / 255.0

# Create a Gaussian filter
h = cv2.GaussianBlur(np.zeros_like(y), (9, 9), sigmaX=0.5)

#out_ADMM = MAP_PnP_ADMM.PlugPlayADMM_super(y,h,K,0.0002)

rows_in,cols_in = y.shape
rows  = np.dot(rows_in, K)
cols  = np.dot(cols_in,K)
xtilde = cv2.resize(y, (rows_in*K, cols_in*K))
out_a = approximal_map.approximal_map_F(h,K,1,y,xtilde)

# Write the restored image
# Convert the denoised image to uint8 if it's of a different data type
out_uint8 = (out_a*255).astype(np.uint8)

# Write the denoised image to a file
cv2.imwrite('RestoredImage.png', out_uint8)
# cv2.imwrite('RestoredImage.png', out)

## Display the resulting image (optional)
# cv2.imshow('Restored Image', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


