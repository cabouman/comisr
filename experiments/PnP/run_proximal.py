import numpy as np
import cv2
# import MAP_PnP_ADMM
import proximal_map as proximal_map
K = 4

y = cv2.imread('DownImage8.png', cv2.IMREAD_GRAYSCALE)

# Convert image to double precision
y = y.astype(np.float64) / 255.0

# Create a Gaussian filter
h = cv2.GaussianBlur(np.zeros_like(y), (9, 9), sigmaX=0.5)

#out_ADMM = MAP_PnP_ADMM.PlugPlayADMM_super(y,h,K,0.0002)

rows_in,cols_in = y.shape
rows  = np.dot(rows_in, K)
cols  = np.dot(cols_in,K)

x = np.zeros((rows_in*K, cols_in*K), dtype=np.float64)

for iter in range(10):
    out_a = proximal_map.proximal_map_F(h,K,1,y,x)

    
    if iter % 2 == 0:
        # Convert the denoised image to uint8 if it's of a different data type
        out_uint8 = (out_a*255).astype(np.uint8)

        # Write the denoised image to a file
        output_filename = ('../data/output/proximal_{:03d}.png'.format(iter+1))  # Adjust file extension as needed
    
        cv2.imwrite(output_filename, out_uint8)



