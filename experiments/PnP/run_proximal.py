import numpy as np
import cv2
# import MAP_PnP_ADMM
import proximal_map as proximal_map
from scipy.ndimage import gaussian_filter

K = 4

y_in = cv2.imread('DownImage.png', cv2.IMREAD_GRAYSCALE)

# Convert image to double precision
# measured_image = measured_image.astype(np.float64) / 255.0

# Check if the image was successfully read
if y_in is not None:
    # Convert image to double precision
    y = y_in.astype(np.float64) / 255.0

else:
    print("Error: Failed to read the image")
    
# Create a Gaussian filter
    
""" 
filter_psf = cv2.GaussianBlur(np.zeros_like(measured_image), (9, 9), sigmaX=0.5)

# Assuming measured_image is defined elsewhere or you have its shape available
measured_image = np.zeros((100, 100))  # Example shape, adjust as needed

# Create a Gaussian filter with a kernel size of (9, 9) and standard deviation sigmaX=0.5
filter_psf = cv2.GaussianBlur(np.zeros_like(measured_image), (9, 9), sigmaX=5)

# Calculate and print the maximum value of the resulting Gaussian filter
print(np.max(filter_psf)) """

# Define the kernel size and standard deviation
kernel_size = (9, 9)
sigmaX = 0.05

h = proximal_map.create_rotationally_symmetric_gaussian_filter(kernel_size, sigmaX)
print("Rotationally symmetric Gaussian filter:")
print(h)


#out_ADMM = MAP_PnP_ADMM.PlugPlayADMM_super(measured_image,filter_psf,subsampling_rate,0.0002)

rows_in,cols_in = y.shape
rows  = np.dot(rows_in, K)
cols  = np.dot(cols_in, K)

x = np.zeros((rows_in*K, cols_in*K), dtype=np.float64)
# x = cv2.imread('OrigImage.png', cv2.IMREAD_GRAYSCALE)
# Check if the image was successfully read
if x is not None:
    # Convert image to double precision
    x = x.astype(np.float64) / 255.0
else:
    print("Error: Failed to read the original image")

for iter in range(100):

    out_a = proximal_map.proximal_map_F(x,h,K,1,y)
    
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

        print("mean of output and diff value:", "\t", out_uint8.mean(), "\t", diff_uint8.mean())

    x = out_a
print ("--------")
print("mean of input image value:", "\t", y_in.mean())




