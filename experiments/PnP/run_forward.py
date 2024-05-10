import numpy as np
import cv2
import proximal_map
K = 4

# Prepare the blurred image
# Read the high-resolution image
z = cv2.imread('../image/input/OrigImage.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('OrigImage.png', z)


# Check if the image was successfully read
if z is not None:
    # Convert image to double precision
    z = z.astype(np.float64) / 255.0
else:
    print("Error: Failed to read the image")


# Create a Gaussian filter
# Define the kernel size and standard deviation
kernel_size = (9, 9)
sigmaX = 0.05

h = proximal_map.create_rotationally_symmetric_gaussian_filter(kernel_size, sigmaX)
print("Rotationally symmetric Gaussian filter:")
print(h)

# Downsample the image
y = cv2.resize(z, (z.shape[1]//K, z.shape[0]//K))

# Apply the filter
filtered_image = cv2.filter2D(y, -1, h)

# Convert the denoised image to uint8 if it's of a different image type
out_uint8 = (filtered_image*255).astype(np.uint8)
cv2.imwrite('DownImage.png', out_uint8)

""" 
# Add Gaussian noise
noise_level = 10 / 255.0
np.random.seed(0)  # Set random seed for reproducibility
noise = noise_level * np.random.randn(*measured_image.shape)
measured_image += noise 

# Write the denoised image to a file
cv2.imwrite('NoisyImage.png', measured_image*255) """



