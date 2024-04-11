import numpy as np
import cv2
# import MAP_PnP_ADMM
import proximal_map as proximal_map
from scipy.ndimage import gaussian_filter

K = 4

y = cv2.imread('DownImage8.png', cv2.IMREAD_GRAYSCALE)

# Convert image to double precision
# y = y.astype(np.float64) / 255.0

# Check if the image was successfully read
if y is not None:
    # Convert image to double precision
    y = y.astype(np.float64) / 255.0
else:
    print("Error: Failed to read the image")
    
# Create a Gaussian filter
    
""" 
h = cv2.GaussianBlur(np.zeros_like(y), (9, 9), sigmaX=0.5)

# Assuming y is defined elsewhere or you have its shape available
y = np.zeros((100, 100))  # Example shape, adjust as needed

# Create a Gaussian filter with a kernel size of (9, 9) and standard deviation sigmaX=0.5
h = cv2.GaussianBlur(np.zeros_like(y), (9, 9), sigmaX=5)

# Calculate and print the maximum value of the resulting Gaussian filter
print(np.max(h)) """

# Define the kernel size and standard deviation
kernel_size = (9, 9)
sigmaX = 0.5

h = proximal_map.create_rotationally_symmetric_gaussian_filter(kernel_size, sigmaX)
print("Rotationally symmetric Gaussian filter:")
print(h)


#out_ADMM = MAP_PnP_ADMM.PlugPlayADMM_super(y,h,K,0.0002)

rows_in,cols_in = y.shape
rows  = np.dot(rows_in, K)
cols  = np.dot(cols_in,K)

x = np.zeros((rows_in*K, cols_in*K), dtype=np.float64)

for iter in range(100):

    out_a = proximal_map.proximal_map_F(h,K,1,y,x)
    
    if iter % 1 == 0:
        # Convert the denoised image to uint8 if it's of a different data type
        out_uint8 = (out_a*255).astype(np.uint8)
        diff_uint8 = ((out_a-x)*255).astype(np.uint8)

        # Write the denoised image to a file
        output_filename = ('../data/output/proximal_{:03d}.png'.format(iter+1))  # Adjust file extension as needed
        diff_filename = ('../data/output/diff_{:03d}.png'.format(iter+1))  # Adjust file extension as needed
        #output_filename = ('proximal_{:03d}.png'.format(iter+1))  # Adjust file extension as needed

        cv2.imwrite(output_filename, out_uint8)
        cv2.imwrite(diff_filename, diff_uint8)

    x = out_a





