import numpy as np
import cv2
import MAP_PnP_ADMM
import proximal_map as proximal_map

# Load in the noisy image
y = cv2.imread('NoisyImage.png', cv2.IMREAD_GRAYSCALE)

# set the scalling factor 
K = 4

# Convert image to double precision
y = y.astype(np.float64) / 255.0

# Create a Gaussian filter
h = cv2.GaussianBlur(np.zeros_like(y), (9, 9), sigmaX=0.5)

out_ADMM = MAP_PnP_ADMM.PlugPlayADMM_super(y,h,K,0.0002)

# Write the restored image
# Convert the denoised image to uint8 if it's of a different data type
out_uint8 = (out_ADMM*255).astype(np.uint8)

# Write the denoised image to a file
cv2.imwrite('RestoredImage_PnP.png', out_uint8)
# cv2.imwrite('RestoredImage.png', out)

## Display the resulting image (optional)
# cv2.imshow('Restored Image', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


